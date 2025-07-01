"""
Reinforcement Learning agent for Pikken AI using Stable Baselines3.
"""

from typing import Any, Dict, Optional
import numpy as np
from .base_agent import BaseAgent

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False


class RLAgent(BaseAgent):
    """
    Reinforcement Learning agent using PPO algorithm.
    Can be trained through self-play against other agents.
    """
    
    def __init__(self, name: str = "RLBot", model_path: Optional[str] = None):
        super().__init__(name)
        self.model = None
        self.model_path = model_path
        
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable-baselines3 is required for RLAgent")
        
        if model_path:
            self.load_model(model_path)
    
    def decide_action(self, observation: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """
        Use trained model to decide action.
        
        Args:
            observation: Current game state (structured dict)
            info: Game information
            
        Returns:
            Model-predicted action
        """
        if self.model is None:
            # Fallback to random action if no model loaded
            import random
            return random.randint(0, 41)
        
        # Convert structured observation to format expected by model
        if isinstance(observation, dict):
            # Flatten the structured observation for the model
            obs_array = self._flatten_observation(observation)
        else:
            obs_array = observation
        
        action, _ = self.model.predict(obs_array, deterministic=False)
        return int(action)
    
    def _flatten_observation(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Flatten structured observation dict into a single array for the model.
        Must match the format used during training (exactly 27 dimensions).
        """
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
    
    def create_model(self, env, **kwargs):
        """Create a new PPO model."""
        default_params = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'verbose': 1
        }
        default_params.update(kwargs)
        
        self.model = PPO('MlpPolicy', env, **default_params)
        return self.model
    
    def train(self, total_timesteps: int):
        """Train the model."""
        if self.model is None:
            raise ValueError("No model created. Call create_model first.")
        
        self.model.learn(total_timesteps=total_timesteps)
    
    def save_model(self, path: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(path)
        self.model_path = path
    
    def load_model(self, path: str):
        """Load a pre-trained model."""
        self.model = PPO.load(path)
        self.model_path = path
