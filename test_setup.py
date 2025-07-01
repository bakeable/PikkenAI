#!/usr/bin/env python3
"""
Test script to verify Pikken AI installation and basic functionality.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        from pikken_env import PikkenEnv
        print("✓ PikkenEnv imported successfully")
    except ImportError as e:
        print(f"✗ PikkenEnv import failed: {e}")
        return False
    
    try:
        from agents.random_agent import RandomAgent
        from agents.heuristic_agent import HeuristicAgent
        print("✓ Agent classes imported successfully")
    except ImportError as e:
        print(f"✗ Agent import failed: {e}")
        return False
    
    # Optional imports
    try:
        import gymnasium as gym
        print("✓ gymnasium available")
    except ImportError:
        print("⚠ gymnasium not available (install with: pip install gymnasium)")
    
    try:
        import stable_baselines3
        print("✓ stable-baselines3 available")
    except ImportError:
        print("⚠ stable-baselines3 not available (install with: pip install stable-baselines3)")
    
    try:
        import matplotlib
        print("✓ matplotlib available")
    except ImportError:
        print("⚠ matplotlib not available (install with: pip install matplotlib)")
    
    return True

def test_environment():
    """Test basic environment functionality."""
    print("\nTesting environment...")
    
    try:
        from pikken_env import PikkenEnv
        from agents.random_agent import RandomAgent
        
        # Create environment and agent
        env = PikkenEnv(num_players=4)
        agent = RandomAgent()
        
        # Test reset
        observation, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Observation shape: {observation.shape}")
        print(f"  Players alive: {info['players_alive']}")
        
        # Test a few steps
        for step in range(5):
            action = agent.decide_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"  Game ended after {step + 1} steps")
                break
        
        print("✓ Environment step functionality working")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_agents():
    """Test different agent types."""
    print("\nTesting agents...")
    
    try:
        from agents.random_agent import RandomAgent
        from agents.heuristic_agent import HeuristicAgent
        import numpy as np
        
        # Create test observation
        obs = np.array([1, 2, 3, 4, 5, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        info = {'current_bid': (2, 3), 'players_alive': [True, True, True, True]}
        
        # Test random agent
        random_agent = RandomAgent()
        action = random_agent.decide_action(obs, info)
        print(f"✓ RandomAgent action: {action}")
        
        # Test heuristic agent
        heuristic_agent = HeuristicAgent()
        action = heuristic_agent.decide_action(obs, info)
        print(f"✓ HeuristicAgent action: {action}")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        return False

def test_tournament():
    """Test tournament functionality."""
    print("\nTesting tournament...")
    
    try:
        from evaluate import Tournament
        from agents.random_agent import RandomAgent
        from agents.heuristic_agent import HeuristicAgent
        
        agents = [
            RandomAgent("Random1"),
            RandomAgent("Random2"),
            HeuristicAgent("Heuristic1"),
            HeuristicAgent("Heuristic2")
        ]
        
        tournament = Tournament(agents, num_games=10)
        results = tournament.run_tournament(verbose=False)
        
        print(f"✓ Tournament completed")
        print(f"  Games played: {results['tournament_info']['games_played']}")
        print(f"  Agents tested: {len(results['agent_stats'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Tournament test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Pikken AI Test Suite ===\n")
    
    tests = [
        test_imports,
        test_environment,
        test_agents,
        test_tournament
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Pikken AI is ready to use.")
        return 0
    else:
        print("⚠ Some tests failed. Check dependencies and installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
