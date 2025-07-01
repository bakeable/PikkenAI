#!/usr/bin/env python3
"""
Quick test script to verify action masking works correctly
"""

from pikken_env import PikkenEnv
from agents.random_agent import RandomAgent

def test_action_masking():
    """Test that action masking prevents infinite loops."""
    print("Testing action masking...")
    
    env = PikkenEnv(num_players=3)
    obs, info = env.reset()
    
    print(f"Initial state:")
    print(f"  Current bid: {env.current_bid}")
    print(f"  Current player: {env.current_player}")
    print(f"  Valid actions: {info['valid_actions']}")
    
    # Ensure call bluff (action 0) is NOT in valid actions at start
    if 0 in info['valid_actions'] and env.current_bid == (0, 0):
        print("❌ ERROR: Call bluff (action 0) should not be valid when no bid exists!")
        return False
    
    # Test a few steps with random agents
    agents = [RandomAgent(f"Test{i}") for i in range(3)]
    
    for step in range(10):
        current_player = env.current_player
        agent = agents[current_player]
        
        # Get current player's observation and info
        player_obs = obs.get(current_player, obs) if isinstance(obs, dict) else obs
        player_info = info.get(current_player, info) if isinstance(info, dict) else info
        
        print(f"\nStep {step}: Player {current_player}")
        print(f"  Current bid: {env.current_bid}")
        print(f"  Valid actions: {player_info['valid_actions']}")
        
        # Verify action 0 is only valid when there's a bid to challenge
        if env.current_bid == (0, 0) and 0 in player_info['valid_actions']:
            print("❌ ERROR: Call bluff should not be valid when no bid exists!")
            return False
        
        try:
            action = agent.decide_action(player_obs, player_info)
            print(f"  Chosen action: {action}")
            
            obs, rewards, terminated, info = env.step(action)
            
            if any(terminated.values() if isinstance(terminated, dict) else [terminated]):
                print("Game ended")
                break
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return False
    
    print("✅ Action masking test passed!")
    return True

if __name__ == "__main__":
    test_action_masking()
