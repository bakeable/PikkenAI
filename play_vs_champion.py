#!/usr/bin/env python3
"""
Human vs Champion Console Game

This script allows a human player to play Pikken against the champion AI model
in the console. The human player will see their dice and game state, then input
their moves which get converted to actions in the PikkenEnv.
"""

import os
import sys
import numpy as np
from typing import Optional, Tuple, Dict, Any

from pikken_env import PikkenEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.rl_agent import RLAgent


class HumanAgent:
    """Agent that prompts human player for input."""
    
    def __init__(self, name: str = "Human"):
        self.name = name
    
    def decide_action(self, observation: Dict[str, Any], info: Dict[str, Any]) -> int:
        """Get action from human input."""
        return self._get_human_input(observation, info)
    
    def _get_human_input(self, observation: Dict[str, Any], info: Dict[str, Any]) -> int:
        """Display game state and get human input."""
        print("\n" + "="*60)
        print(f"🎲 YOUR TURN ({self.name})")
        print("="*60)
        
        # Display game state
        self._display_game_state(observation, info)
        
        # Get current bid for context
        current_bids = observation.get('current_bids', np.zeros((12, 3)))
        current_bid = None
        if len(current_bids) > 0 and current_bids[0][0] > 0:
            current_bid = (int(current_bids[0][1]), int(current_bids[0][2]))
        
        # Simple user interface
        while True:
            try:
                # Ask if they want to call bluff
                if current_bid:
                    print(f"\n🚨 Current bid to beat: {current_bid[0]}x {current_bid[1]}'s")
                    bluff_choice = input("Do you want to call bluff? (y/n): ").strip().lower()
                    
                    if bluff_choice in ['y', 'yes']:
                        return 0  # Call bluff action
                    elif bluff_choice not in ['n', 'no']:
                        print("❌ Please enter 'y' for yes or 'n' for no")
                        continue
                else:
                    print("\n📝 No previous bid - you'll make the first bid")
                
                # Get bid from user
                return self._get_bid_from_user(current_bid)
                
            except KeyboardInterrupt:
                print("\n🚪 Exiting game...")
                sys.exit(0)
    
    def _get_bid_from_user(self, current_bid: Optional[Tuple[int, int]]) -> int:
        """Get a bid from user input and convert to action."""
        while True:
            try:
                # Get quantity
                quantity_input = input("\nEnter quantity (number of dice): ").strip()
                if not quantity_input.isdigit():
                    print("❌ Please enter a valid number for quantity")
                    continue
                quantity = int(quantity_input)
                
                if quantity < 1:
                    print("❌ Quantity must be at least 1")
                    continue
                if quantity > 20:
                    print("❌ Quantity seems too high (max 20)")
                    continue
                
                # Get face value
                face_input = input("Enter face value (1-6): ").strip()
                if not face_input.isdigit():
                    print("❌ Please enter a valid number for face value")
                    continue
                face_value = int(face_input)
                
                if face_value < 1 or face_value > 6:
                    print("❌ Face value must be between 1 and 6")
                    continue
                
                # Validate bid is higher than current bid
                if current_bid:
                    curr_qty, curr_face = current_bid
                    is_higher = (quantity > curr_qty) or (quantity == curr_qty and face_value > curr_face)
                    if not is_higher:
                        print(f"❌ Your bid ({quantity}x {face_value}'s) must be higher than current bid ({curr_qty}x {curr_face}'s)")
                        print("   Either increase quantity, or keep same quantity with higher face value")
                        continue
                
                # Convert to action index
                action = self._convert_bid_to_action(quantity, face_value)
                print(f"✅ Making bid: {quantity}x {face_value}'s")
                return action
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"❌ Error: {e}. Please try again.")
                continue
    
    def _convert_bid_to_action(self, quantity: int, face_value: int) -> int:
        """Convert quantity and face value to action index."""
        # Simple mapping: action = 1 + (quantity-1)*6 + (face_value-1)
        # This ensures each (quantity, face_value) pair maps to a unique action
        action = 1 + (quantity - 1) * 6 + (face_value - 1)
        
        # Ensure action is within valid range (1-41, since 0 is call bluff)
        action = max(1, min(41, action))
        return action
    
    def _display_game_state(self, observation: Dict[str, Any], info: Dict[str, Any]):
        """Display current game state."""
        global_state = observation.get('global_state', np.zeros(5))
        own_dice = observation.get('own_dice', np.zeros(5))
        player_statuses = observation.get('player_statuses', np.zeros((4, 2)))
        current_bids = observation.get('current_bids', np.zeros((12, 3)))
        
        # Display total dice count
        total_dice = int(global_state[0]) if len(global_state) > 0 else 0
        print(f"🎲 Total dice in game: {total_dice}")
        
        # Display your dice
        print(f"\n🎯 Your dice:")
        dice_display = []
        for i, face_value in enumerate(own_dice):
            dice_display.append(f"{face_value}")
        
        if dice_display:
            print(f"   {' | '.join(dice_display)}")
        else:
            print("   No dice remaining")
        
        # Display player statuses
        print(f"\n👥 Players alive:")
        for i in range(len(player_statuses)):
            alive = bool(player_statuses[i][0])
            dice_count = int(player_statuses[i][1])
            status = "🟢" if alive else "💀"
            if i == 0:
                print(f"   Player {i+1} (YOU): {status} {dice_count} dice")
            else:
                print(f"   Player {i+1}: {status} {dice_count} dice")
        
        # Display recent bids
        print(f"\n📋 Recent bids:")
        valid_bids = []
        for i in range(min(3, len(current_bids))):
            bid_data = current_bids[i]
            if len(bid_data) >= 3 and bid_data[0] > 0:  # Valid bid
                player_id = int(bid_data[0]) - 1  # Convert to 0-based
                quantity = int(bid_data[1])
                face_value = int(bid_data[2])
                valid_bids.append((player_id, quantity, face_value))
        
        if valid_bids:
            for player_id, quantity, face_value in valid_bids:
                print(f"   Player {player_id+1}: {quantity}x{face_value}")
        else:
            print("   No bids yet")


def display_game_intro():
    """Display game introduction and rules."""
    print("🎲" + "="*58 + "🎲")
    print("          PIKKEN AI - HUMAN VS CHAMPION")
    print("🎲" + "="*58 + "🎲")
    print()
    print("🎯 GAME RULES:")
    print("   • Each player starts with 5 dice")
    print("   • Dice show values 1-6, with 1s as 'wild' (count as any value)")
    print("   • Players bid on total dice across ALL players")
    print("   • Each bid must be higher than the previous (quantity or value)")
    print("   • Call 'bluff' if you think the current bid is impossible")
    print("   • If bluff is correct: bidder loses a die")
    print("   • If bluff is wrong: challenger loses a die")
    print("   • Last player with dice wins!")
    print()
    print("🤖 OPPONENTS:")
    print("   • Champion AI: Trained with advanced self-play")
    print("   • Heuristic AI: Strategic rule-based player")
    print("   • Random AI: Unpredictable moves")
    print()


def select_opponents() -> list:
    """Allow user to select opponent configuration."""
    print("🎯 SELECT OPPONENTS:")
    print("1. Champion AI only (1v1)")
    print("2. Champion + Heuristic AI (1v2)")
    print("3. Champion + 2 Heuristic AIs (1v3)")
    print("4. Mixed: Champion + Heuristic + Random (1v3)")
    
    while True:
        try:
            choice = input("\nChoose opponent setup (1-4): ").strip()
            choice_num = int(choice)
            
            if choice_num == 1:
                return [load_champion_agent()]
            elif choice_num == 2:
                return [
                    load_champion_agent(),
                    HeuristicAgent("Heuristic_Strategic", 0.6)
                ]
            elif choice_num == 3:
                return [
                    load_champion_agent(),
                    HeuristicAgent("Heuristic_Conservative", 0.4),
                    HeuristicAgent("Heuristic_Aggressive", 0.8)
                ]
            elif choice_num == 4:
                return [
                    load_champion_agent(),
                    HeuristicAgent("Heuristic_Balanced", 0.5),
                    RandomAgent("Random_Chaos")
                ]
            else:
                print("❌ Please choose 1-4")
                
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n🚪 Exiting...")
            sys.exit(0)


def load_champion_agent() -> RLAgent:
    """Load the champion AI model."""
    # Try different possible champion model paths
    possible_paths = [
        "./models/selfplay_champion.zip",
        "./models/selfplay_champion_best/best_model.zip",
        "./models/pikken_agent.zip",
        "./models/pikken_agent_best/best_model.zip"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                print(f"✅ Loading champion from: {path}")
                return RLAgent("Champion_AI", path)
            except Exception as e:
                print(f"⚠️ Failed to load {path}: {e}")
                continue
    
    # Fallback to heuristic if no RL model found
    print("⚠️ No champion model found, using advanced heuristic agent")
    return HeuristicAgent("Champion_Heuristic", 0.7)


def play_game(opponents: list) -> Optional[int]:
    """Play a single game and return winner (0 = human, others = AIs)."""
    # Create all agents
    agents = [HumanAgent("Human")] + opponents
    num_players = len(agents)
    
    # Create environment
    env = PikkenEnv(num_players=num_players)
    observation, info = env.reset()
    
    print(f"\n🎮 Starting game with {num_players} players!")
    print(f"   You are Player 1")
    for i, agent in enumerate(opponents, 1):
        print(f"   Player {i+1}: {agent.name}")
    
    game_over = False
    step_count = 0
    max_steps = 500  # Prevent infinite games
    
    while not game_over and step_count < max_steps:
        current_player = env.current_player
        current_agent = agents[current_player]
        
        # Get observation for current player
        if isinstance(observation, dict):
            current_obs = observation.get(current_player, observation)
            current_info = info.get(current_player, info)
        else:
            current_obs = observation
            current_info = info
        
        # Display current player
        if current_player != 0:  # Not human
            print(f"\n🤖 {current_agent.name}'s turn...")
        
        try:
            # Get action from current agent
            action = current_agent.decide_action(current_obs, current_info)
            
            # Execute action
            observation, rewards, terminated, info = env.step(action)
            
            # Display AI action
            if current_player != 0:
                action_description = get_action_description(action, current_obs)
                print(f"   {current_agent.name} chose: {action_description}")
            
            # Check for game end
            alive_count = sum(env.players_alive)
            if alive_count <= 1:
                game_over = True
                winner = next((i for i, alive in enumerate(env.players_alive) if alive), -1)
                return winner
                
            # Check terminated condition
            if isinstance(terminated, dict):
                if any(terminated.values()):
                    game_over = True
            else:
                if terminated:
                    game_over = True
                    
        except Exception as e:
            print(f"❌ Error during game: {e}")
            return None
        
        step_count += 1
    
    if step_count >= max_steps:
        print("⏰ Game reached maximum steps, ending...")
        
    return None


def get_action_description(action: int, observation: Dict[str, Any]) -> str:
    """Convert action number to human-readable description."""
    if action == 0:
        return "🚨 Call bluff!"
    
    # Calculate bid from action
    action_id = action - 1
    if action_id < 0:
        return "Unknown action"
    
    quantity = (action_id // 6) + 1
    face_value = (action_id % 6) + 1
    
    return f"📈 Bid {quantity}x{face_value}"


def main():
    """Main game loop."""
    display_game_intro()
    
    # Select opponents
    opponents = select_opponents()
    
    # Game statistics
    games_played = 0
    human_wins = 0
    
    print(f"\n🎮 Ready to play! Press Ctrl+C anytime to quit.")
    
    while True:
        games_played += 1
        print(f"\n🎯 GAME {games_played}")
        print("-" * 40)
        
        try:
            winner = play_game(opponents)
            
            if winner is not None:
                if winner == 0:
                    print(f"\n🎉 CONGRATULATIONS! You won Game {games_played}!")
                    human_wins += 1
                else:
                    agent_name = opponents[winner-1].name if winner-1 < len(opponents) else f"Player {winner+1}"
                    print(f"\n💀 Game Over! {agent_name} won Game {games_played}")
                
                # Display statistics
                win_rate = (human_wins / games_played) * 100
                print(f"\n📊 Your Stats: {human_wins}/{games_played} wins ({win_rate:.1f}%)")
            else:
                print(f"\n❌ Game {games_played} ended unexpectedly")
            
            # Ask to play again
            print(f"\n🎮 Play another game?")
            choice = input("Press Enter to continue, or 'q' to quit: ").strip().lower()
            if choice in ['q', 'quit', 'exit']:
                break
                
        except KeyboardInterrupt:
            print(f"\n🚪 Thanks for playing!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            choice = input("Try again? (y/n): ").strip().lower()
            if choice not in ['y', 'yes']:
                break
    
    # Final statistics
    if games_played > 0:
        final_win_rate = (human_wins / games_played) * 100
        print(f"\n📈 FINAL STATS:")
        print(f"   Games played: {games_played}")
        print(f"   Wins: {human_wins}")
        print(f"   Win rate: {final_win_rate:.1f}%")
        
        if final_win_rate >= 50:
            print(f"🏆 Excellent performance! You beat the AI!")
        elif final_win_rate >= 25:
            print(f"💪 Good job! You held your own against the AI!")
        else:
            print(f"🤖 The AI dominated, but keep practicing!")
    
    print(f"\n👋 Thanks for playing Pikken AI!")


if __name__ == "__main__":
    main()
