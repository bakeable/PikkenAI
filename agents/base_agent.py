"""
Base agent class for Pikken AI players.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all Pikken AI agents.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.games_played = 0
        self.games_won = 0
    
    @abstractmethod
    def decide_action(self, observation: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """
        Decide what action to take given the current observation.
        
        Args:
            observation: Current game state observation (structured dict)
            info: Additional game information
            
        Returns:
            Action index (0 for call bluff, 1+ for bids)
        """
        pass
    
    def update_stats(self, won: bool):
        """Update agent statistics."""
        self.games_played += 1
        if won:
            self.games_won += 1
    
    def get_win_rate(self) -> float:
        """Get current win rate."""
        if self.games_played == 0:
            return 0.0
        return self.games_won / self.games_played
    
    def reset_stats(self):
        """Reset statistics."""
        self.games_played = 0
        self.games_won = 0
