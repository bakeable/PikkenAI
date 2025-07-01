#!/usr/bin/env python3
"""
Sandbox script to demonstrate the PikkenEnv in action.

This script runs a game between a HeuristicAgent and RandomAgent,
showing step-by-step gameplay with detailed console output.
"""

import time
import numpy as np
from typing import Dict, Any

from constants import MAX_ACTIONS
from pikken_env import PikkenEnv
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent


def print_separator(title: str = ""):
    """Print a visual separator with optional title."""
    print("\n" + "="*60)
    if title:
        print(f" {title} ".center(60, "="))
        print("="*60)
    else:
        print()


def print_observation(obs: Dict[str, np.ndarray], player_id: int):
    """Print formatted observation for a player."""
    print(f"\n--- Player {player_id} Observation ---")
    
    global_state = obs.get('global_state', np.zeros(5))
    own_dice = obs.get('own_dice', np.zeros(5))
    player_statuses = obs.get('player_statuses', np.zeros((6, 2)))
    current_bids = obs.get('current_bids', np.full((12, 3), -1))
    
    # Global state
    round_num = int(global_state[0]) if len(global_state) > 0 else 0
    turn_num = int(global_state[1]) if len(global_state) > 1 else 0
    current_player = int(global_state[2]) if len(global_state) > 2 else 0
    phase = "Supporting" if int(global_state[3]) == 1 else "Bidding"
    bluff_target = int(global_state[4]) if len(global_state) > 4 else -1
    
    print(f"  Round: {round_num}, Turn: {turn_num}")
    print(f"  Phase: {phase}")
    print(f"  Current Player: {current_player}")
    if bluff_target != -1:
        print(f"  Bluff Target: {bluff_target}")
    
    # Own dice
    own_dice_list = [int(die) for die in own_dice if die > 0]
    print(f"  Own Dice: {own_dice_list}")
    
    # Player statuses
    print("  Player Status:")
    for i in range(len(player_statuses)):
        if i < 6:  # Max players
            dice_count = int(player_statuses[i][1])
            if dice_count > 0:
                print(f"    Player {i}: {dice_count} dice")
    
    # Recent bids
    print("  Recent Bids:")
    for i, bid in enumerate(current_bids):
        if bid[0] != -1:  # Valid bid
            player, qty, face = int(bid[0]), int(bid[1]), int(bid[2])
            print(f"    Player {player}: {qty} x {face}s")
        if i >= 5:  # Show only last 5 bids
            break


def print_action_decision(agent_name: str, action: int, observation: Dict, info: Dict):
    """Print the agent's action decision with context."""
    print(f"\n🎯 {agent_name} Decision:")
    
    current_bid = info.get('current_bid', (0, 0))
    global_state = observation.get('global_state', np.zeros(5))
    phase = "Supporting" if int(global_state[3]) == 1 else "Bidding"
    
    if phase == "Supporting":
        if action == 0:
            print(f"   Action {action}: Support the BIDDER")
        elif action == 1:
            print(f"   Action {action}: Support the CHALLENGER")
        else:
            print(f"   Action {action}: Invalid support action!")
    else:
        if action == 0:
            print(f"   Action {action}: Call BLUFF on bid {current_bid}")
        else:
            # Convert action to bid
            bid_index = action - 1
            quantity = (bid_index // 6) + 1
            face_value = (bid_index % 6) + 1
            print(f"   Action {action}: Bid {quantity} x {face_value}s")


def print_game_state(env: PikkenEnv):
    """Print current game state."""
    print(f"\n🎲 Game State:")
    print(f"   Round: {env.round_count}")
    print(f"   Phase: {env.phase}")
    print(f"   Current Player: {env.current_player}")
    print(f"   Current Bid: {env.current_bid}")
    
    print(f"   Players & Dice:")
    for i, (alive, dice) in enumerate(zip(env.players_alive, env.players_dice)):
        status = "ALIVE" if alive else "DEAD"
        dice_str = str(dice) if dice else "[]"
        print(f"     Player {i}: {dice_str} ({status})")
    
    if env.phase == "supporting":
        print(f"   Bluff Challenge: Player {env.bluff_target['bidder']} vs Player {env.bluff_target['challenger']}")
        print(f"   Support Votes: {env.support_votes}")


def convert_multi_agent_to_single(observations: Dict, rewards: Dict, terminations: Dict, 
                                 infos: Dict, current_player: int):
    """Convert multi-agent format to single agent format for the current player."""
    return (
        observations.get(current_player, {}),
        rewards.get(current_player, 0.0),
        terminations.get(current_player, False),
        infos.get(current_player, {})
    )


def run_sandbox_game():
    """Run a complete demonstration game."""
    print_separator("PIKKEN AI SANDBOX DEMO")
    
    # Create environment and agents
    env = PikkenEnv(num_players=4, max_dice=5)
    
    agents = [
        HeuristicAgent("Heuristic-Conservative", aggressiveness=0.3),
        RandomAgent("Random-1"),
        HeuristicAgent("Heuristic-Aggressive", aggressiveness=0.8),
        RandomAgent("Random-2")
    ]
    
    print("🤖 Agents:")
    for i, agent in enumerate(agents):
        print(f"   Player {i}: {agent.name}")
    
    # Reset environment
    print_separator("GAME START")
    observation, info = env.reset()
    
    step_count = 0
    max_steps = 100  # Prevent infinite games
    
    print_game_state(env)
    
    while step_count < max_steps:
        current_player_id = env.current_player
        
        # Check if game is over
        alive_count = sum(env.players_alive)
        if alive_count <= 1:
            print_separator("GAME OVER")
            winner = None
            for i, alive in enumerate(env.players_alive):
                if alive:
                    winner = i
                    break
            
            if winner is not None:
                print(f"🏆 Winner: Player {winner} ({agents[winner].name})")
            else:
                print("🤝 No winner (draw)")
            break
        
        print_separator(f"STEP {step_count + 1}")
        print(f"👤 Current Player: {current_player_id} ({agents[current_player_id].name})")
        
        # Convert multi-agent observation to single-agent for current player
        current_obs = observation.get(current_player_id, {})
        current_info = info.get(current_player_id, {})
        
        # Print observation
        print_observation(current_obs, current_player_id)
        
        # Get agent action
        agent = agents[current_player_id]
        invalid_action = True
        retries = 0
        while invalid_action:
            try:
                action = agent.decide_action(current_obs, current_info)
                print_action_decision(agent.name, action, current_obs, current_info)
            except Exception as e:
                print(f"❌ Error getting action from {agent.name}: {e}")
                action = 0  # Default to call bluff
            
            # Execute step
            print(f"\n⚡ Executing action {action}...")
            retries += 1
            observations, rewards, terminations, infos = env.step(action)
            invalid_action = infos.get(current_player_id, {}).get("invalid_action", False)
            
            if invalid_action:
                print(f"⚠️ Invalid action by {agent.name}, retrying...")
                if retries >= MAX_ACTIONS:
                    raise f"❗ Too many invalid actions"

        
        # Update for next iteration
        observation = observations
        info = infos
        
        # Print results
        current_reward = rewards.get(current_player_id, 0.0)
        if current_reward != 0:
            print(f"💰 Reward for Player {current_player_id}: {current_reward}")
        
        # Print updated game state
        print_game_state(env)
        
        # Check for termination
        if any(terminations.values()):
            print("\n🏁 Game terminated!")
            break
        
        step_count += 1
        
        # Small delay for readability
        time.sleep(2)
    
    if step_count >= max_steps:
        print_separator("GAME TIMEOUT")
        print("⏰ Game reached maximum steps limit")
    
    # Final statistics
    print_separator("FINAL STATISTICS")
    for i, agent in enumerate(agents):
        status = "ALIVE" if env.players_alive[i] else "ELIMINATED"
        dice_count = len(env.players_dice[i])
        print(f"Player {i} ({agent.name}): {status}, {dice_count} dice remaining")


def main():
    """Main function."""
    print("Welcome to the Pikken AI Sandbox!")
    print("This demonstration shows the environment and agents in action.")
    
    try:
        run_sandbox_game()
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Sandbox demo complete!")


if __name__ == "__main__":
    main()
