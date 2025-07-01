#!/usr/bin/env python3
"""
Advanced Self-Play Training for Pikken AI

This script implements sophisticated self-play training where multiple RL agents
compete against each other in different configurations to create superior players.
"""

import os
import argparse
import numpy as np
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import gymnasium as gym
from copy import deepcopy

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not available. Install with: pip install stable-baselines3")
    STABLE_BASELINES_AVAILABLE = False

from pikken_env import PikkenEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.rl_agent import RLAgent


class SelfPlayWrapper(gym.Env):
    """
    Advanced self-play wrapper where RL agents train against multiple opponents
    including other RL agents, creating an evolutionary training environment.
    """
    
    def __init__(self, base_env: PikkenEnv, agent_pool: List, current_generation: int = 0):
        super().__init__()
        self.base_env = base_env
        self.agent_pool = agent_pool  # Pool of available opponents
        self.current_generation = current_generation
        self.current_opponents = []
        
        # Copy observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(27,), dtype=np.float32
        )
        self.action_space = base_env.action_space
        
    def reset(self, **kwargs):
        # Select diverse opponents for this game
        self.current_opponents = self._select_opponents()
        
        obs, info = self.base_env.reset(**kwargs)
        
        # Return observation for RL agent (player 0)
        rl_obs = obs if isinstance(obs, dict) else obs.get(0, obs)
        rl_info = info if isinstance(info, dict) else info.get(0, info)
        
        # Handle any opponent turns that might come first
        while self.base_env.current_player != 0 and sum(self.base_env.players_alive) > 1:
            opponent_obs, opponent_info = self._get_current_player_obs(obs, info)
            opponent_action = self._get_opponent_action(opponent_obs, opponent_info)
            obs, rewards, terminated, info = self.base_env.step(opponent_action)
            
            if any(terminated.values() if isinstance(terminated, dict) else [terminated]):
                break
        
        return self._flatten_obs(rl_obs), rl_info
    
    def step(self, action):
        # Execute RL agent's action
        obs, rewards, terminated, info = self.base_env.step(action)
        
        # Get RL agent's reward and termination status
        rl_reward = rewards.get(0, 0.0) if isinstance(rewards, dict) else rewards
        rl_terminated = terminated.get(0, False) if isinstance(terminated, dict) else terminated
        rl_truncated = False
        rl_obs = obs.get(0, obs) if isinstance(obs, dict) else obs
        rl_info = info.get(0, info) if isinstance(info, dict) else info
        
        # Handle opponent turns
        while (not rl_terminated and not rl_truncated and 
               self.base_env.current_player != 0 and 
               sum(self.base_env.players_alive) > 1):
            
            opponent_obs, opponent_info = self._get_current_player_obs(obs, info)
            opponent_action = self._get_opponent_action(opponent_obs, opponent_info)
            obs, rewards, terminated, info = self.base_env.step(opponent_action)
            
            # Update RL agent's observation
            rl_obs = obs.get(0, obs) if isinstance(obs, dict) else obs
            rl_info = info.get(0, info) if isinstance(info, dict) else info
            
            # Check if game ended
            if any(terminated.values() if isinstance(terminated, dict) else [terminated]):
                rl_terminated = terminated.get(0, False) if isinstance(terminated, dict) else terminated
                # Add final reward if RL agent won
                if sum(self.base_env.players_alive) <= 1 and self.base_env.players_alive[0]:
                    rl_reward += 10.0
                break
        
        return self._flatten_obs(rl_obs), rl_reward, rl_terminated, rl_truncated, rl_info
    
    def _select_opponents(self) -> List:
        """Select diverse opponents for training."""
        num_opponents = self.base_env.num_players - 1
        
        # Probability distribution for opponent selection
        if self.current_generation == 0:
            # Early training: mostly heuristic and random agents
            opponent_weights = {
                'random': 0.3,
                'heuristic': 0.6,
                'rl_current': 0.1,
                'rl_previous': 0.0
            }
        elif self.current_generation < 5:
            # Mid training: mix of all types
            opponent_weights = {
                'random': 0.2,
                'heuristic': 0.3,
                'rl_current': 0.3,
                'rl_previous': 0.2
            }
        else:
            # Advanced training: mostly RL agents
            opponent_weights = {
                'random': 0.1,
                'heuristic': 0.2,
                'rl_current': 0.4,
                'rl_previous': 0.3
            }
        
        selected_opponents = []
        for _ in range(num_opponents):
            opponent_type = np.random.choice(
                list(opponent_weights.keys()),
                p=list(opponent_weights.values())
            )
            
            if opponent_type == 'random':
                selected_opponents.append(RandomAgent(f"Random_Gen{self.current_generation}"))
            elif opponent_type == 'heuristic':
                aggressiveness = np.random.uniform(0.2, 0.8)
                selected_opponents.append(HeuristicAgent(f"Heuristic_Gen{self.current_generation}", aggressiveness))
            elif opponent_type == 'rl_current' and len(self.agent_pool) > 0:
                # Select from current generation RL agents
                current_agents = [a for a in self.agent_pool if 'current' in a.name]
                if current_agents:
                    selected_opponents.append(np.random.choice(current_agents))
                else:
                    selected_opponents.append(HeuristicAgent(f"Heuristic_Fallback_Gen{self.current_generation}", 0.5))
            elif opponent_type == 'rl_previous' and len(self.agent_pool) > 1:
                # Select from previous generation RL agents
                previous_agents = [a for a in self.agent_pool if 'previous' in a.name]
                if previous_agents:
                    selected_opponents.append(np.random.choice(previous_agents))
                else:
                    selected_opponents.append(HeuristicAgent(f"Heuristic_Fallback_Gen{self.current_generation}", 0.5))
            else:
                # Fallback to heuristic
                selected_opponents.append(HeuristicAgent(f"Heuristic_Fallback_Gen{self.current_generation}", 0.5))
        
        return selected_opponents
    
    def _get_current_player_obs(self, obs, info):
        """Get observation and info for current player."""
        current_player = self.base_env.current_player
        player_obs = obs.get(current_player, obs) if isinstance(obs, dict) else obs
        player_info = info.get(current_player, info) if isinstance(info, dict) else info
        return player_obs, player_info
    
    def _get_opponent_action(self, obs, info):
        """Get action from appropriate opponent agent."""
        current_player = self.base_env.current_player
        if current_player == 0:
            return 0  # Shouldn't happen
        
        opponent_idx = min(current_player - 1, len(self.current_opponents) - 1)
        opponent = self.current_opponents[opponent_idx]
        
        try:
            return opponent.decide_action(obs, info)
        except Exception as e:
            print(f"Warning: Opponent {opponent.name} error: {e}")
            return 0  # Default to call bluff
    
    def _flatten_obs(self, obs_dict):
        """Flatten structured observation into array."""
        if isinstance(obs_dict, dict):
            flattened_parts = []
            
            # Global state (5 values)
            global_state = obs_dict.get('global_state', np.zeros(5))
            flattened_parts.append(global_state.astype(np.float32))
            
            # Own dice (5 values) 
            own_dice = obs_dict.get('own_dice', np.zeros(5))
            flattened_parts.append(own_dice.astype(np.float32))
            
            # Player statuses (8 values: 4 players x 2 features)
            player_statuses = obs_dict.get('player_statuses', np.zeros((4, 2)))
            flattened_parts.append(player_statuses.flatten()[:8].astype(np.float32))
            
            # Current bids (9 values: 3 most recent bids x 3 features)
            current_bids = obs_dict.get('current_bids', np.zeros((12, 3)))
            recent_bids = current_bids[:3].flatten()[:9]
            flattened_parts.append(recent_bids.astype(np.float32))
            
            # Combine all parts (should be exactly 27 values)
            result = np.concatenate(flattened_parts)
            # Ensure exactly 27 features
            if len(result) < 27:
                result = np.pad(result, (0, 27 - len(result)), 'constant')
            elif len(result) > 27:
                result = result[:27]
                
            return result
        else:
            # Already flattened
            return np.array(obs_dict, dtype=np.float32)
    
    def render(self, mode='human'):
        return self.base_env.render(mode)
    
    def __getattr__(self, name):
        return getattr(self.base_env, name)


class SelfPlayCallback(BaseCallback):
    """Callback to manage self-play progression and agent pool updates."""
    
    def __init__(self, agent_pool: List, generation_steps: int = 25000, verbose=0):
        super().__init__(verbose)
        self.agent_pool = agent_pool
        self.generation_steps = generation_steps
        self.current_generation = 0
        self.steps_in_generation = 0
        
    def _on_step(self) -> bool:
        self.steps_in_generation += 1
        
        # Check if it's time to evolve to next generation
        if self.steps_in_generation >= self.generation_steps:
            self._evolve_generation()
            self.steps_in_generation = 0
            self.current_generation += 1
            
        return True
    
    def _evolve_generation(self):
        """Evolve to the next generation by updating agent pool."""
        if self.verbose > 0:
            print(f"\n🧬 Evolving to Generation {self.current_generation + 1}")
        
        # Save current model as previous generation
        current_model_path = f"./models/selfplay_gen_{self.current_generation}.zip"
        os.makedirs(os.path.dirname(current_model_path), exist_ok=True)
        self.model.save(current_model_path)
        
        # Add current model to agent pool as previous generation
        try:
            previous_agent = RLAgent(f"RL_previous_gen_{self.current_generation}", current_model_path)
            self.agent_pool.append(previous_agent)
            
            if self.verbose > 0:
                print(f"✅ Added Gen {self.current_generation} agent to pool")
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Failed to add agent to pool: {e}")
        
        # Keep pool size manageable (max 10 agents)
        if len(self.agent_pool) > 10:
            self.agent_pool.pop(0)  # Remove oldest agent


def create_selfplay_environment(agent_pool: List, generation: int = 0) -> DummyVecEnv:
    """Create vectorized self-play environment."""
    
    def make_env():
        base_env = PikkenEnv(num_players=4)
        return SelfPlayWrapper(base_env, agent_pool, generation)
    
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    
    return env


def train_selfplay_agent(
    total_timesteps: int = 200_000,
    generation_steps: int = 25_000,
    save_path: str = "./models/selfplay_champion",
    eval_freq: int = 10_000,
    checkpoint_freq: int = 25_000,
    initial_model_path: Optional[str] = None
):
    """
    Train an RL agent using advanced self-play techniques.
    
    Args:
        total_timesteps: Total training timesteps
        generation_steps: Steps per generation before evolving
        save_path: Path to save final champion model
        eval_freq: Evaluation frequency
        checkpoint_freq: Checkpoint frequency
        initial_model_path: Path to initial model (optional)
    """
    
    if not STABLE_BASELINES_AVAILABLE:
        print("Error: stable-baselines3 is required for self-play training.")
        return None
    
    print("🚀 Starting Advanced Self-Play Training")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Generation steps: {generation_steps:,}")
    print(f"   Generations planned: {total_timesteps // generation_steps}")
    
    # Initialize agent pool
    agent_pool = []
    
    # Add initial agents if provided
    if initial_model_path and os.path.exists(initial_model_path):
        try:
            initial_agent = RLAgent("RL_initial", initial_model_path)
            agent_pool.append(initial_agent)
            print(f"✅ Loaded initial model: {initial_model_path}")
        except Exception as e:
            print(f"⚠️ Failed to load initial model: {e}")
    
    # Create training environment
    print("Creating self-play training environment...")
    env = create_selfplay_environment(agent_pool, generation=0)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = create_selfplay_environment(agent_pool, generation=0)
    
    # Initialize PPO model
    print("Initializing PPO model...")
    if initial_model_path and os.path.exists(initial_model_path):
        try:
            model = PPO.load(initial_model_path, env=env)
            print(f"✅ Loaded existing model from {initial_model_path}")
        except:
            # If loading fails, create new model
            model = PPO(
                'MlpPolicy',
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                tensorboard_log="./tensorboard_logs/"
            )
    else:
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )
    
    # Create callbacks
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    selfplay_callback = SelfPlayCallback(
        agent_pool=agent_pool,
        generation_steps=generation_steps,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path + "_best",
        log_path="./eval_logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./checkpoints/",
        name_prefix="selfplay_agent"
    )
    
    # Start training
    print(f"🎯 Starting self-play training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[selfplay_callback, eval_callback, checkpoint_callback],
        tb_log_name="selfplay_training"
    )
    
    print(f"💾 Saving champion model to {save_path}")
    model.save(save_path)
    
    # Final evaluation
    print("🏆 Training complete! Running final evaluation...")
    final_agent = RLAgent("Champion", save_path + ".zip")
    
    return model, agent_pool


def evaluate_selfplay_champion(model_path: str, num_games: int = 50):
    """Evaluate the self-play trained champion against various opponents."""
    
    print(f"🧪 Evaluating Self-Play Champion: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Create champion agent
    try:
        champion = RLAgent("SelfPlay_Champion", model_path)
        print("✅ Champion loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load champion: {e}")
        return
    
    # Test against different opponent compositions
    test_configurations = [
        {
            "name": "vs_Random_Agents",
            "agents": [
                champion,
                RandomAgent("Random_1"),
                RandomAgent("Random_2"),
                RandomAgent("Random_3")
            ]
        },
        {
            "name": "vs_Heuristic_Agents",
            "agents": [
                champion,
                HeuristicAgent("Heuristic_Conservative", 0.3),
                HeuristicAgent("Heuristic_Balanced", 0.5),
                HeuristicAgent("Heuristic_Aggressive", 0.8)
            ]
        },
        {
            "name": "vs_Mixed_Agents",
            "agents": [
                champion,
                RandomAgent("Random_1"),
                HeuristicAgent("Heuristic_1", 0.5),
                HeuristicAgent("Heuristic_2", 0.7)
            ]
        }
    ]
    
    results = {}
    
    for config in test_configurations:
        print(f"\n🎯 Testing: {config['name']}")
        agents = config['agents']
        wins = {i: 0 for i in range(len(agents))}
        
        for game_num in range(num_games):
            if game_num % 10 == 0:
                print(f"   Game {game_num + 1}/{num_games}")
            
            env = PikkenEnv(num_players=len(agents))
            observation, info = env.reset()
            
            game_over = False
            step_count = 0
            max_steps = 200
            
            while not game_over and step_count < max_steps:
                current_player = env.current_player
                current_agent = agents[current_player]
                
                # Get observation and info for current player
                current_obs = observation.get(current_player, observation)
                current_info = info.get(current_player, info)
                
                try:
                    action = current_agent.decide_action(current_obs, current_info)
                    observation, rewards, terminated, info = env.step(action)
                    
                    # Check for game end
                    alive_count = sum(env.players_alive)
                    if alive_count <= 1:
                        game_over = True
                        winner = next((i for i, alive in enumerate(env.players_alive) if alive), -1)
                        if winner >= 0:
                            wins[winner] += 1
                        break
                    
                    if any(terminated.values() if isinstance(terminated, dict) else [terminated]):
                        game_over = True
                        
                except Exception as e:
                    print(f"   ❌ Error in game: {e}")
                    break
                
                step_count += 1
        
        # Calculate and store results
        champion_wins = wins[0]
        champion_winrate = champion_wins / num_games * 100
        results[config['name']] = champion_winrate
        
        print(f"   🏆 Champion wins: {champion_wins}/{num_games} ({champion_winrate:.1f}%)")
        for i, agent in enumerate(agents[1:], 1):
            print(f"      {agent.name}: {wins[i]}/{num_games} ({wins[i]/num_games*100:.1f}%)")
    
    # Overall summary
    print(f"\n📊 Champion Overall Performance:")
    avg_winrate = sum(results.values()) / len(results)
    print(f"   Average win rate: {avg_winrate:.1f}%")
    for test_name, winrate in results.items():
        print(f"   {test_name}: {winrate:.1f}%")
    
    return results


def main():
    """Main self-play training function."""
    parser = argparse.ArgumentParser(description="Advanced Self-Play Training for Pikken AI")
    parser.add_argument(
        "--timesteps", type=int, default=200_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--generation-steps", type=int, default=25_000,
        help="Timesteps per generation"
    )
    parser.add_argument(
        "--save-path", type=str, default="./models/selfplay_champion",
        help="Path to save champion model"
    )
    parser.add_argument(
        "--initial-model", type=str, default=None,
        help="Path to initial model to start from"
    )
    parser.add_argument(
        "--eval-games", type=int, default=50,
        help="Number of games for final evaluation"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only evaluate existing model, don't train"
    )
    
    args = parser.parse_args()
    
    if args.eval_only:
        model_path = args.save_path + ".zip"
        evaluate_selfplay_champion(model_path, args.eval_games)
    else:
        print("🧬 === Advanced Self-Play Training ===")
        print(f"Total timesteps: {args.timesteps:,}")
        print(f"Generation steps: {args.generation_steps:,}")
        print(f"Planned generations: {args.timesteps // args.generation_steps}")
        print(f"Save path: {args.save_path}")
        
        # Train the champion
        model, agent_pool = train_selfplay_agent(
            total_timesteps=args.timesteps,
            generation_steps=args.generation_steps,
            save_path=args.save_path,
            initial_model_path=args.initial_model
        )
        
        if model:
            print("\n🏆 Self-play training complete!")
            
            # Evaluate the champion
            champion_path = args.save_path + ".zip"
            if os.path.exists(champion_path):
                evaluate_selfplay_champion(champion_path, args.eval_games)
            
            print(f"\n✨ Champion model saved at: {champion_path}")
            print(f"📈 Training logs available in: ./tensorboard_logs/")
            print(f"🔍 View with: tensorboard --logdir ./tensorboard_logs/")


if __name__ == "__main__":
    main()
