"""
Heuristic agent for Pikken AI - uses simple game logic and probability.
"""

from typing import Any, Dict
import numpy as np

from constants import MAX_ACTIONS
from .base_agent import BaseAgent


class HeuristicAgent(BaseAgent):
    """
    Agent that uses heuristic strategies:
    - Calculates probability of bids being true
    - Considers own dice when making decisions
    - Uses conservative vs aggressive strategies
    """
    
    def __init__(self, name: str = "HeuristicBot", aggressiveness: float = 0.5):
        super().__init__(name)
        self.aggressiveness = aggressiveness  # 0.0 = conservative, 1.0 = aggressive
    
    def decide_action(self, observation: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """
        Make decision based on heuristic analysis.
        
        Args:
            observation: Current game state (structured dict)
            info: Game information
            
        Returns:
            Strategically chosen action
        """
        # Extract observation components
        global_state = observation.get('global_state', np.zeros(5))
        own_dice = observation.get('own_dice', np.zeros(5))
        current_bids = observation.get('current_bids', np.full((12, 3), -1))
        
        # Get game state info
        phase_flag = global_state[3] if len(global_state) > 3 else 0
        current_bid = info.get('current_bid', (0, 0))
        players_alive = sum(info.get('players_alive', [True] * 4))
        
        # Remove zero dice (lost dice) and convert to list
        own_dice_list = [int(die) for die in own_dice if die > 0]
        
        # Support phase - decide who to support
        if phase_flag == 1:
            return self._decide_support(current_bid, own_dice_list, players_alive)
        
        # Bidding phase
        if current_bid == (0, 0):
            # First move - make conservative bid based on own dice
            return self._make_opening_bid(own_dice_list)
        
        # Analyze current bid probability
        bid_probability = self._calculate_bid_probability(current_bid, own_dice_list, players_alive)
        
        # Decision making based on probability and aggressiveness
        if bid_probability < 0.3 - (self.aggressiveness * 0.2):
            # Low probability - call bluff
            return 0
        elif bid_probability > 0.7 + (self.aggressiveness * 0.2):
            # High probability - raise bid
            return self._make_raise_bid(current_bid, own_dice_list)
        else:
            # Medium probability - raise bid based on aggressiveness
            if np.random.random() < self.aggressiveness:
                return self._make_raise_bid(current_bid, own_dice_list)
            else:
                return 0  # call bluff
    
    def _make_opening_bid(self, own_dice: list) -> int:
        """Make opening bid based on own dice."""
        if not own_dice:
            return 1  # Minimum bid (1, 1) -> action 1
        
        # Count most frequent die face (excluding PIK for opening)
        face_counts = {}
        for die in own_dice:
            if die != 1:  # Don't count PIK for opening bid
                face_counts[die] = face_counts.get(die, 0) + 1
        
        if not face_counts:
            # Only have PIKs, bid conservatively
            pik_count = sum(1 for die in own_dice if die == 1)
            return self._convert_bid_to_action((pik_count, 1))
        
        best_face = max(face_counts.keys(), key=lambda x: face_counts[x])
        count = face_counts[best_face]
        
        # Conservative bid: claim what we actually have
        return self._convert_bid_to_action((count, best_face))
    
    def _make_raise_bid(self, current_bid: tuple, own_dice: list) -> int:
        """Make a raise bid."""
        curr_qty, curr_face = current_bid
        
        # Count how many of current face we have (including PIKs for non-PIK faces)
        if curr_face == 1:  # Current bid is PIK
            own_count = sum(1 for die in own_dice if die == 1)
        else:
            own_count = sum(1 for die in own_dice if die == curr_face or die == 1)
        
        # Strategy: raise quantity if we have some, raise face if we don't
        if own_count > 0:
            # Try to raise quantity
            new_qty = curr_qty + 1
            new_face = curr_face
        else:
            # Try to raise face value
            if curr_face < 6:
                new_qty = curr_qty
                new_face = curr_face + 1
            else:
                # Can't raise face, must raise quantity
                new_qty = curr_qty + 1
                new_face = curr_face
        
        # Handle PIK rules
        if curr_face == 1 and new_face != 1:
            # Transitioning from PIK to non-PIK requires doubling + 1
            new_qty = 2 * curr_qty + 1
        
        return self._convert_bid_to_action((new_qty, new_face))
    
    def _calculate_bid_probability(self, bid: tuple, own_dice: list, players_alive: int) -> float:
        """
        Calculate probability that the current bid is truthful.
        
        Uses simplified probability based on:
        - Number of dice in play
        - Own dice
        - Typical distribution assumptions
        """
        if bid == (0, 0):
            return 1.0
        
        quantity, face_value = bid
        
        # Estimate total dice in play (assume average of 3 dice per player)
        estimated_total_dice = players_alive * 3
        
        # Count how many of this face we have
        own_count = sum(1 for die in own_dice if die == face_value)
        
        # Need to find (quantity - own_count) dice of face_value among other players
        needed_from_others = max(0, quantity - own_count)
        other_dice = estimated_total_dice - len(own_dice)
        
        if other_dice <= 0:
            return 1.0 if needed_from_others == 0 else 0.0
        
        # Probability that at least needed_from_others dice show face_value
        # Using simplified binomial approximation
        expected_others = other_dice / 6.0  # Expected number of any face
        
        if needed_from_others == 0:
            return 1.0
        elif needed_from_others > other_dice:
            return 0.0
        else:
            # Rough probability calculation
            prob = max(0.0, min(1.0, expected_others / needed_from_others))
            return prob
    
    def _decide_support(self, current_bid: tuple, own_dice: list, players_alive: int) -> int:
        """Decide whether to support bidder or challenger."""
        # Calculate if bid seems truthful based on own dice
        bid_probability = self._calculate_bid_probability(current_bid, own_dice, players_alive)
        
        # Support bidder if probability is high, challenger if low
        if bid_probability > 0.6:
            return 0  # Support bidder
        else:
            return 1  # Support challenger
    
    def _convert_bid_to_action(self, bid: tuple) -> int:
        """Convert (quantity, face_value) bid to action index."""
        quantity, face_value = bid
        
        # Action 0 is call bluff, actions 1+ are bids
        # Formula: action = 1 + (quantity-1)*6 + (face_value-1)
        action = 1 + (quantity - 1) * 6 + (face_value - 1)
        
        # Cap at reasonable maximum (30 * 6 = 180 total actions)
        return min(action, MAX_ACTIONS)
