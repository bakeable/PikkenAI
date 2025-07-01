"""
Training script for Pikken AI using reinforcement learning.

This script trains an RL agent to play Pikken through self-play
against various opponents including random and heuristic agents.
"""

import os
import argparse
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import gymnasium as gym

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not available. Install with: pip install stable-baselines3")
    STABLE_BASELINES_AVAILABLE = False

from pikken_env import PikkenEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.rl_agent import RLAgent


class SingleAgentWrapper(gym.Env):
    """
    Wrapper to convert multi-agent PikkenEnv to single-agent for RL training.
    The RL agent is always player 0, and opponents handle other players.
    """
    
    def __init__(self, base_env: PikkenEnv, opponents: List):
        super().__init__()
        self.base_env = base_env
        self.opponents = opponents
        self.current_opponents = []
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space
        
    def reset(self, **kwargs):
        # Randomly select opponents for this game
        num_opponents = self.base_env.num_players - 1
        self.current_opponents = np.random.choice(
            self.opponents, size=num_opponents, replace=True
        ).tolist()
        
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
        
        return rl_obs, rl_info
    
    def step(self, action):
        # Execute RL agent's action
        obs, rewards, terminated, info = self.base_env.step(action)
        
        # Get RL agent's reward and termination status
        rl_reward = rewards.get(0, 0.0) if isinstance(rewards, dict) else rewards
        rl_terminated = terminated.get(0, False) if isinstance(terminated, dict) else terminated
        rl_truncated = False  # PikkenEnv doesn't use truncated
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
        
        return rl_obs, rl_reward, rl_terminated, rl_truncated, rl_info
    
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
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.base_env.render(mode)
    
    def __getattr__(self, name):
        """Delegate other attributes to base environment."""
        return getattr(self.base_env, name)


def create_training_environment(num_players: int = 4) -> DummyVecEnv:
    """Create vectorized environment for training."""
    
    # Create opponent agents
    opponents = [
        HeuristicAgent("Heuristic1", aggressiveness=0.25),
        HeuristicAgent("Heuristic2", aggressiveness=0.4),
        HeuristicAgent("Heuristic3", aggressiveness=0.6),
        HeuristicAgent("Heuristic4", aggressiveness=0.75),
    ]
    
    def make_env():
        base_env = PikkenEnv(num_players=num_players)
        wrapped_env = SingleAgentWrapper(base_env, opponents)
        # Convert to simple observation space for stable-baselines3
        return FlattenObservationWrapper(wrapped_env)
    
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    
    return env


class FlattenObservationWrapper(gym.Env):
    """Wrapper to flatten structured observations for stable-baselines3."""
    
    def __init__(self, env):
        super().__init__()
        self.env = env
        # Create flattened observation space
        # Global(5) + Own dice(5) + Player status(8) + Current bids(9) = 27 features
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-100, high=100, shape=(27,), dtype=np.float32
        )
        self.action_space = env.action_space
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info
    
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
        return self.env.render(mode)
    
    def __getattr__(self, name):
        return getattr(self.env, name)


def train_agent(
    total_timesteps: int = 100_000,
    save_path: str = "./models/pikken_agent",
    eval_freq: int = 10_000,
    checkpoint_freq: int = 25_000
):
    """
    Train an RL agent to play Pikken.
    
    Args:
        total_timesteps: Number of training steps
        save_path: Path to save the trained model
        eval_freq: How often to evaluate the agent
        checkpoint_freq: How often to save checkpoints
    """
    
    if not STABLE_BASELINES_AVAILABLE:
        print("Error: stable-baselines3 is required for training.")
        print("Install with: pip install stable-baselines3")
        return None
    
    print("Creating training environment...")
    env = create_training_environment()
    
    print("Creating evaluation environment...")
    eval_env = create_training_environment()
    
    print("Initializing PPO agent...")
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
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path + "_best",
        log_path="./eval_logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=20
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./checkpoints/",
        name_prefix="pikken_agent"
    )
    
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="pikken_training"
    )
    
    print(f"Saving final model to {save_path}")
    model.save(save_path)
    
    return model


def plot_training_progress(log_dir: str = "./eval_logs/"):
    """Plot training progress from evaluation logs."""
    try:
        import pandas as pd
        
        # Load evaluation results
        eval_file = os.path.join(log_dir, "evaluations.npz")
        if os.path.exists(eval_file):
            data = np.load(eval_file)
            timesteps = data['timesteps']
            results = data['results']
            
            # Plot results
            plt.figure(figsize=(10, 6))
            mean_rewards = np.mean(results, axis=1)
            std_rewards = np.std(results, axis=1)
            
            plt.plot(timesteps, mean_rewards, label='Mean Reward')
            plt.fill_between(
                timesteps,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.3
            )
            
            plt.xlabel('Timesteps')
            plt.ylabel('Episode Reward')
            plt.title('Pikken AI Training Progress')
            plt.legend()
            plt.grid(True)
            plt.savefig('training_progress.png')
            plt.show()
        else:
            print(f"No evaluation file found at {eval_file}")
    
    except ImportError:
        print("matplotlib and pandas required for plotting")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Pikken AI agent")
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Number of training timesteps"
    )
    parser.add_argument(
        "--save-path", type=str, default="./models/pikken_agent",
        help="Path to save trained model"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=10_000,
        help="Evaluation frequency"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Plot training progress after training"
    )
    
    args = parser.parse_args()
    
    print("=== Pikken AI Training ===")
    print(f"Training timesteps: {args.timesteps}")
    print(f"Save path: {args.save_path}")
    print(f"Evaluation frequency: {args.eval_freq}")
    
    # Train the agent
    model = train_agent(
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        eval_freq=args.eval_freq
    )
    
    if model and args.plot:
        print("Plotting training progress...")
        plot_training_progress()
    
    print("Training complete!")


if __name__ == "__main__":
    main()
