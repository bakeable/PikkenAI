#!/usr/bin/env python3
"""
Test script to evaluate the trained RL agent.
"""

import os
from pikken_env import PikkenEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.rl_agent import RLAgent

def test_trained_agent(model_path: str = "./models/pikken_agent.zip", num_games: int = 5):
    """Test the trained RL agent against other agents."""
    
    print(f"🧪 Testing trained RL agent from {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Create agents
    try:
        rl_agent = RLAgent("TrainedRL", model_path=model_path)
        print("✅ RL agent loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load RL agent: {e}")
        return
    
    agents = [
        rl_agent,
        rl_agent,
        HeuristicAgent("Heuristic1", aggressiveness=0.25),
        HeuristicAgent("Heuristic2", aggressiveness=0.5),
        HeuristicAgent("Heuristic3", aggressiveness=0.75),
    ]
    
    # Test games
    wins = {i: 0 for i in range(5)}
    
    for game_num in range(num_games):
        print(f"\n🎮 Game {game_num + 1}/{num_games}")
        
        env = PikkenEnv(num_players=4)
        observation, info = env.reset()
        
        game_over = False
        step_count = 0
        max_steps = 20000
        
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
                        print(f"   Winner: Player {winner} ({agents[winner].name})")
                    else:
                        print("   No winner (draw)")
                    break
                
                if any(terminated.values() if isinstance(terminated, dict) else [terminated]):
                    game_over = True
                    
            except Exception as e:
                print(f"   ❌ Error in game {game_num + 1}: {e}")
                break
            
            step_count += 1
        
        if step_count >= max_steps:
            print(f"   ⏰ Game {game_num + 1} reached max steps")
    
    # Print results
    print(f"\n📊 Results after {num_games} games:")
    for i, agent in enumerate(agents):
        win_rate = wins[i] / num_games * 100
        print(f"   Player {i} ({agent.name}): {wins[i]} wins ({win_rate:.1f}%)")

if __name__ == "__main__":
    test_trained_agent()
