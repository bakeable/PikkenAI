"""
PikkenEnv - Gymnasium Environment for the Pikken Dice Game

This module implements a gym-compatible environment for the Pikken game,
where players bluff and bid with dice.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random

from constants import MAX_ACTIONS, MAX_DICE, MAX_FACE, MAX_PLAYERS


class PikkenEnv(gym.Env):
    """
    Gymnasium environment for the Pikken dice game.
    
    Game Rules:
    - Each player has 5 dice
    - Players take turns bidding on dice combinations across all players
    - A bid consists of (quantity, face_value) - e.g., "three 4s"
    - Next player must either raise the bid or call bluff
    - If bluff is called, dice are revealed and loser loses a die
    - Last player with dice wins
    
    Action Space:
        Bidding phase:
        - 0: Call bluff (or support bid)
        - >=1: Make bid (quantity, face_value combinations)

        Support phase:
        - 0: Support bidder
        - 1: Support challenger
        - >1: Invalid actions
        
    Observation Space:
    - Own dice (5 values, 0 if die is lost)
    - Current bid (quantity, face_value)
    - Number of players still in game
    - Player position
    - Bid history features
    """
    
    def __init__(self, num_players: int = MAX_PLAYERS, max_dice: int = MAX_DICE):
        super().__init__()
        
        self.num_players = num_players
        self.max_dice = max_dice
        self.current_player = 0
        self.players_dice = []
        self.players_alive = []
        
        # Game state
        self.current_bid = (0, 0)  # (quantity, face_value)
        self.bid_history = []
        self.structured_bid_history = []
        self.round_count = 0
        
        self.phase = "bidding"
        self.support_votes = {}
        self.bluff_target = {"bidder": -1, "challenger": -1}
        
        # Action space: pass, call_bluff, raise_bid(20 combinations), bluff_bid(20 combinations)
        self.total_bid_actions = MAX_PLAYERS * MAX_DICE
        self.action_space = spaces.Discrete(1 + self.total_bid_actions * MAX_FACE) # 1 for call bluff, other for bidding actions
        
        # Observation space (structured)
        self.observation_space = spaces.Dict({
            "global_state": spaces.Box(low=0, high=100, shape=(5,), dtype=np.int32),
            "own_dice": spaces.Box(low=0, high=6, shape=(MAX_DICE,), dtype=np.int32),
            "player_statuses": spaces.Box(low=0, high=10, shape=(MAX_PLAYERS, 2), dtype=np.int32),
            "current_bids": spaces.Box(low=0, high=12, shape=(12, 3), dtype=np.int32),  # memory of 12 bids per round
            "past_bids": spaces.Box(low=0, high=100, shape=(100, 5), dtype=np.int32)    # max 100 historical bids
        })
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize players with full dice
        self.players_dice = []
        self.players_alive = []
        
        for i in range(self.num_players):
            dice = self._roll_dice(self.max_dice)
            self.players_dice.append(dice)
            self.players_alive.append(True)
        
        self.current_player = self.bluff_target["challenger"] if self.bluff_target["challenger"] != -1 else 0
        self.current_bid = (0, 0)
        self.bid_history = []
        self.structured_bid_history = []
        self.round_count = 0
        
        self.phase = "bidding"
        self.support_votes = {}
        self.bluff_target = {"bidder": -1, "challenger": -1}
        self._prev_dice_count = {i: len(self.players_dice[i]) for i in range(self.num_players)}
        self._consecutive_invalid = {i: 0 for i in range(self.num_players)}
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def finish_round(self):
        """Finish the current round and reset state."""
        print(f"\n=== Round {self.round_count} Finished ===")
        # Remove dice based on bluff call outcome
        if self._is_valid_bid(self.current_bid):
            # Bid was truthful, bidder and supporters lose a die
            for player in self.support_votes:
                if self.support_votes[player] == "bidder":
                    print(f"Player {player} was correct in bid, removing die")
                    self._remove_die(player)
        else:
            # Bid was wrong, challenger and supporters lose a die
            for player in self.support_votes:
                if self.support_votes[player] == "challenger":
                    print(f"Player {player} was correct in challenge, removing die")
                    self._remove_die(player)

        # Reset round state
        self.current_player = self.bluff_target["challenger"]
        self.current_bid = (0, 0)
        self.round_count += 1
        self.phase = "bidding"
        self.support_votes = {}
        self.bluff_target = {"bidder": -1, "challenger": -1}
        print(f"\nStarting Round {self.round_count} with {self.num_players} players and {self._get_dice_left()} dice left...")

    
    def step(self, action: int) -> Tuple[Dict[int, Any], Dict[int, float], Dict[int, bool], Dict[int, Dict]]:
        """
        Execute one step in the environment. Multi-agent compatible.
        Only the current player's action is processed.
        """
        reward, step_valid, terminated = 0.0, False, False

        agent_id = self.current_player

        if not self.players_alive[agent_id]:
            self._next_player()
            observations = {agent: self._get_observation() for agent in range(self.num_players)}
            rewards = {agent: 0.0 for agent in range(self.num_players)}
            terminations = {agent: False for agent in range(self.num_players)}
            infos = {agent: self._get_info() for agent in range(self.num_players)}
            return observations, rewards, terminations, infos

        if self.phase == "supporting":
            reward, step_valid = self._handle_support(action)
        else:
            if action == 0:
                reward, step_valid = self._handle_call_bluff()
            else:
                reward, step_valid = self._handle_bid(action)

        # Handle invalid actions
        if not step_valid:
            # Invalid action: give moderate penalty but allow learning
            print(f"Player {agent_id} made invalid action {action}")
            reward = -1.0  # Moderate penalty instead of -10.0
            
            # SAFETY: If this is the 3rd+ consecutive invalid action by this player,
            # force advancement to prevent infinite loops
            if not hasattr(self, '_consecutive_invalid'):
                self._consecutive_invalid = {}
            self._consecutive_invalid[agent_id] = self._consecutive_invalid.get(agent_id, 0) + 1
            
            if self._consecutive_invalid[agent_id] >= 100:
                print(f"SAFETY: Player {agent_id} had 100+ invalid actions, forcing advancement")
                raise ValueError(f"Player {agent_id} made too many invalid actions")
        else:
            # Reset invalid action counter on valid action
            if hasattr(self, '_consecutive_invalid'):
                self._consecutive_invalid[agent_id] = 0

        # Give small positive reward for making valid moves
        if step_valid and not terminated:
            reward += 0.05  # Small reward for valid actions

        if len(self.support_votes) >= sum(self.players_alive):
            self.finish_round()

        # Check for winner: first player to reach 0 dice wins
        winner = None
        for player_idx in range(self.num_players):
            if len(self.players_dice[player_idx]) == 0:
                winner = player_idx
                break
                
        if winner is not None:
            terminated = True
            if agent_id == winner:
                reward += 50.0  # Big reward for winning!
                print(f"Player {agent_id} WINS the game!")
            else:
                reward -= 2.0   # Small penalty for losing
                print(f"Player {agent_id} loses to player {winner}")

        # Reward for losing dice (getting closer to winning in Pikken)
        current_dice = len(self.players_dice[agent_id])
        if hasattr(self, '_prev_dice_count'):
            prev_dice = self._prev_dice_count.get(agent_id, current_dice)
            if current_dice < prev_dice:
                reward += 5.0  # Reward for losing a die (progress toward winning)
                print(f"Player {agent_id} lost a die - closer to winning!")
        
        # Update dice count tracking
        if not hasattr(self, '_prev_dice_count'):
            self._prev_dice_count = {}
        self._prev_dice_count[agent_id] = current_dice

        # Only advance to next player if the action was valid
        if step_valid:
            self._next_player()
        observations = {agent: self._get_observation() for agent in range(self.num_players)}
        rewards = {agent: (reward if agent == agent_id else 0.0) for agent in range(self.num_players)}
        terminations = {agent: terminated for agent in range(self.num_players)}
        infos = {agent: self._get_info(invalid_action=not step_valid) for agent in range(self.num_players)}
        return observations, rewards, terminations, infos
    
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get structured observation for the current player."""
        # Global state
        round_number = self.round_count
        turn_number = len(self.bid_history)
        phase_flag = 0 if self.phase == "bidding" else 1
        bluff_target_idx = self.bluff_target["challenger"]
        global_state = np.array([
            round_number,
            turn_number,
            self.current_player,
            phase_flag,
            bluff_target_idx
        ], dtype=np.int32)

        # Own dice
        own_dice = np.zeros(self.max_dice, dtype=np.int32)
        player_dice = self.players_dice[self.current_player]
        for i, die in enumerate(player_dice):
            if i < self.max_dice:
                own_dice[i] = die

        # Player statuses
        player_statuses = np.zeros((self.num_players, 2), dtype=np.int32)
        for idx in range(self.num_players):
            player_statuses[idx][0] = idx
            player_statuses[idx][1] = sum(1 for die in self.players_dice[idx] if die > 0)

        # Current bids (this round only)
        current_bids = np.full((12, 3), -1, dtype=np.int32)
        for i, bid in enumerate(reversed(self.bid_history[-12:])):
            current_bids[i] = bid

        # Past bids
        past_bids = np.full((100, 5), -1, dtype=np.int32)
        for i, entry in enumerate(self.structured_bid_history[-100:]):
            round_number, turn_number, player_id, qty, face = entry
            past_bids[i] = [round_number, turn_number, player_id, qty, face]

        return {
            "global_state": global_state,
            "own_dice": own_dice,
            "player_statuses": player_statuses,
            "current_bids": current_bids,
            "past_bids": past_bids
        }

    def _get_previous_player_index(self, steps_back: int) -> int:
        """Estimate the player index who made a bid N steps ago (simplified)."""
        idx = (self.current_player - steps_back - 1) % self.num_players
        while not self.players_alive[idx]:
            idx = (idx - 1) % self.num_players
        return idx
    
    def _get_info(self, invalid_action = False) -> Dict:
        """Get additional info about the game state."""
        return {
            'invalid_action': invalid_action,
            'current_player': self.current_player,
            'players_alive': self.players_alive.copy(),
            'current_bid': self.current_bid,
            'round_count': self.round_count,
            'total_dice': sum(sum(1 for die in player_dice if die > 0) for player_dice in self.players_dice),
            'valid_actions': self._get_valid_actions()
        }
    
    
    def _handle_call_bluff(self) -> Tuple[float, bool]:
        """Handle call bluff action."""
        if self.current_bid == (0, 0):
            print(f"Player {self.current_player} tried to call bluff with no bid")
            return -10.0, False  # Can't call bluff on first move
        
        self.phase = "supporting"
        self.bluff_target["bidder"] = self._get_previous_player()
        self.bluff_target["challenger"] = self.current_player
        self.support_votes = {
            self._get_previous_player(): "bidder",
            self.current_player: "challenger",
        }
        print(f"Player {self.current_player} calls bluff on player {self.bluff_target['bidder']} with bid {self.current_bid}")

        return 0.2, True # Slightly better reward for valid call bluff action
    
    def _handle_bid(self, action: int) -> Tuple[float, bool]:
        """Handle bid action."""
        # Convert action to bid
        quantity, face_value = self._action_to_bid(action)
        new_bid = (quantity, face_value)
        print(f"Player {self.current_player} bids: {new_bid}")
        
        # Check if bid is valid (higher than current)
        if not self._is_valid_raise(new_bid):
            print(f"Invalid bid by player {self.current_player}: {new_bid}")
            return -1.0, False  # Invalid bid penalty
        
        self.current_bid = new_bid
        self.bid_history.append((self.current_player, quantity, face_value))
        turn_number = len([b for b in self.structured_bid_history if b[0] == self.round_count])
        self.structured_bid_history.append((self.round_count, turn_number, self.current_player, quantity, face_value))
        return 0.15, True  # Slightly better reward for valid bid action
    
    def _handle_support(self, action: int) -> Tuple[float, bool]:
        """Handle support action during bluff call."""
        if action not in [0, 1]:
            print(f"Player {self.current_player} made invalid support action: {action}")
            return -1.0, False # Invalid support action
        
        self.support_votes[self.current_player] = "bidder" if action == 0 else "challenger"

        # Support bidder if action == 0, else support challenger
        bid_truthful = self._is_valid_bid(self.current_bid)
        support_valid = (action == 0 and bid_truthful) or (action == 1 and not bid_truthful)

        print(f"Player {self.current_player} supports {'bidder' if action == 0 else 'challenger'}")
        return (0.3 if support_valid else -0.5), True # Better reward for correct support, smaller penalty for wrong support
    
    def _is_valid_raise(self, new_bid: Tuple[int, int]) -> bool:
        """Check if new bid is higher than current bid."""
        if new_bid[0] > self._get_dice_left():
            return False  # Can't bid more than total dice left
        
        if self.current_bid == (0, 0):
            return True # Any other bid is valid if no current bid
        
        curr_qty, curr_face = self.current_bid
        new_qty, new_face = new_bid
        
        # If current bid is PIK
        if curr_face == 1:
            # PIK can only be raised by increasing quantity and bidding PIK
            if new_face == 1 and new_qty > curr_qty:
                return True
            
            # Or doubling quantity + 1 with a non-PIK face
            if new_face != 1 and new_qty >= 2 * curr_qty + 1:
                return True
            
            # Otherwise, invalid raise
            return False
        
        # Otherwise, a higher quantity is always valid
        if new_qty > curr_qty:
            return True
        
        # If the same quantity, then higher face value is valid
        if new_qty == curr_qty and new_face > curr_face:
            return True
        
        return False
    
    def _is_valid_bid(self, bid: Tuple[int, int]) -> bool:
        """Check if new bid is valid based on all dice in play."""
        # Count total dice
        dice_map = {}
        for i, alive in enumerate(self.players_alive):
            if alive:
                for die in self.players_dice[i]:
                    if die in dice_map:
                        dice_map[die] += 1
                    else:
                        dice_map[die] = 1
        
        # Check if bid is valid
        quantity, face_value = bid
        if face_value == 1 and quantity <= dice_map.get(1, 0):
            return True
        elif face_value != 1 and quantity <= dice_map.get(face_value, 0) + dice_map.get(1, 0):
            return True
        
        return False
    
    def _roll_dice(self, num_dice: int) -> List[int]:
        """Roll specified number of dice."""
        return [random.randint(1, 6) for _ in range(num_dice)]
    
    
    
    def _remove_die(self, player_idx: int):
        """Remove a die from specified player."""
        if self.players_dice[player_idx]:
            self.players_dice[player_idx].pop()
            print(f"Player {player_idx} now has {len(self.players_dice[player_idx])} dice")
            
        # In Pikken: If player has no dice left, they WIN!
        # Keep them alive until game ends, but they're the winner
        if len(self.players_dice[player_idx]) == 0:
            print(f"Player {player_idx} has reached 0 dice - WINNER!")
            # Don't mark as dead - they won!
    
    def _next_player(self):
        """Move to next alive player."""
        start_player = self.current_player
        self.current_player = (self.current_player + 1) % self.num_players
        
        # Skip dead players
        while not self.players_alive[self.current_player] and self.current_player != start_player:
            self.current_player = (self.current_player + 1) % self.num_players
    
    def _get_previous_player(self) -> int:
        """Get previous alive player."""
        prev_player = (self.current_player - 1) % self.num_players
        while not self.players_alive[prev_player]:
            prev_player = (prev_player - 1) % self.num_players
        return prev_player
    
    def _get_dice_left(self) -> int:
        """Count total dice left across all players."""
        dice_left = 0
        for i, alive in enumerate(self.players_alive):
            if alive:
                for die in self.players_dice[i]:
                    if die > 0:
                        dice_left += 1

        return dice_left
    
    def _get_valid_actions(self) -> list:
        """Get list of valid actions for current player."""
        valid_actions = []
        
        if self.phase == "supporting":
            # During support phase, only support actions are valid
            valid_actions = [0, 1]  # 0 = support bidder, 1 = support challenger
        else:
            # During bidding phase
            # Call bluff is valid ONLY if there's a current bid to challenge
            if self.current_bid != (0, 0):
                valid_actions.append(0)
            
            # Check which bid actions are valid
            for action in range(1, MAX_ACTIONS):
                quantity, face_value = self._action_to_bid(action)
                if quantity > 0 and face_value > 0:  # Valid bid format
                    new_bid = (quantity, face_value)
                    if self._is_valid_raise(new_bid):
                        valid_actions.append(action)
            
            # Ensure we always have at least one valid action
            # If it's the first move and no valid bids, force action 1 (smallest bid)
            if not valid_actions and self.current_bid == (0, 0):
                valid_actions.append(1)  # Force first player to make a bid
            
            # If still no valid actions, allow call bluff as last resort
            if not valid_actions:
                valid_actions.append(0)
            
        return valid_actions
    
    def _action_to_bid(self, action: int) -> Tuple[int, int]:
        """Convert action number to (quantity, face_value) bid."""
        if action == 0:
            return (0, 0)  # Call bluff
        
        # Map action 1-180 to bid combinations
        # Each combination represents (quantity, face_value)
        # where quantity ranges from 1 to MAX_PLAYERS*MAX_DICE (30)
        # and face_value ranges from 1 to 6
        action_idx = action - 1
        max_quantity = MAX_PLAYERS * MAX_DICE
        
        # Calculate quantity (1 to 30) and face_value (1 to 6)
        quantity = (action_idx // MAX_FACE) + 1
        face_value = (action_idx % MAX_FACE) + 1
        
        # Ensure we don't exceed max possible quantity
        if quantity > max_quantity:
            quantity = max_quantity
            
        return (quantity, face_value)
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """Render the current game state."""
        if mode == 'human':
            print(f"\n=== Round {self.round_count} ===")
            print(f"Current Player: {self.current_player}")
            print(f"Current Bid: {self.current_bid}")
            
            for i, (alive, dice) in enumerate(zip(self.players_alive, self.players_dice)):
                status = "ALIVE" if alive else "DEAD"
                print(f"Player {i}: {dice} ({status})")
            
            if self.bid_history:
                print(f"Recent bids: {self.bid_history[-3:]}")
        
        return None
