"""
Random agent for Pikken AI - makes random valid moves.
"""

import random
from typing import Any, Dict
import numpy as np

from constants import MAX_ACTIONS, MAX_BID_ACTIONS, MAX_FACE
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent that makes random valid moves.
    Useful as a baseline for training and evaluation.
    """
    
    def __init__(self, name: str = "RandomBot"):
        super().__init__(name)
    
    def decide_action(self, observation: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """
        Choose a random valid action.
        
        Args:
            observation: Current game state (structured dict)
            info: Game information including current bid
            
        Returns:
            Random valid action
        """
        valid_actions = self.valid_actions(observation, info)
        return random.choice(valid_actions)


    def valid_actions(self, observation: Dict[str, np.ndarray], info: Dict[str, Any]) -> list:
        """
        Get list of valid actions based on current game state.
        
        Args:
            observation: Current game state (structured dict)
            info: Game information including current bid
            
        Returns:
            List of valid action indices
        """
        current_bid = info.get('current_bid', (0, 0))
        global_state = observation.get('global_state', np.zeros(5))
        phase_flag = global_state[3] if len(global_state) > 3 else 0
        
        if phase_flag == 1:
            # Supporting phase - can support bidder (0) or challenger (1)
            return [0, 1]
        
        # Bidding phase
        player_statuses = observation.get('player_statuses', np.zeros((6, 2)))
        dice_left = sum(player_statuses[:, 1])
        if current_bid == (0, 0):
            # No bid yet, must make a bid (action 1+)
            return list(range(1, dice_left * MAX_FACE + 1))
        
        # With existing bid, can call bluff (0) or raise bid (1+)
        valid_actions = [0]  # call_bluff

        # Calculate all possible raise actions
        curr_qty, curr_face = current_bid
        for quantity in range(1, dice_left + 1):
            for face in range(1, MAX_FACE + 1):
                action = (quantity - 1) * MAX_FACE + (face - 1) + 1
                if quantity > curr_qty or (quantity == curr_qty and face > curr_face):
                    valid_actions.append(action)

        return valid_actions
    