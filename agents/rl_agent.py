"""
Reinforcement Learning agent for Pikken AI using Stable Baselines3.
"""

from typing import Any, Dict, Optional
import numpy as np

from constants import MAX_ACTIONS
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
        Use trained model to decide action with action masking.
        
        Args:
            observation: Current game state (structured dict)
            info: Game information including valid_actions
            
        Returns:
            Model-predicted action (guaranteed to be valid)
        """
        if self.model is None:
            # Fallback to random valid action if no model loaded
            valid_actions = info.get('valid_actions', [0])
            import random
            return random.choice(valid_actions)
        
        # Convert structured observation to format expected by model
        if isinstance(observation, dict):
            # Flatten the structured observation for the model
            obs_array = self._flatten_observation(observation)
        else:
            obs_array = observation
        
        # Get valid actions from info
        valid_actions = info.get('valid_actions', list(range(MAX_ACTIONS)))
        
        # Use action masking to only select valid actions
        return self._predict_with_action_mask(obs_array, valid_actions)
    
    def _predict_with_action_mask(self, obs_array: np.ndarray, valid_actions: list) -> int:
        """
        Predict action using the model with action masking.
        
        Args:
            obs_array: Flattened observation
            valid_actions: List of valid action indices
            
        Returns:
            Valid action selected by the model
        """
        if len(valid_actions) == 1:
            # Only one valid action, return it
            return valid_actions[0]
        
        try:
            import torch
            
            # Convert numpy array to torch tensor with correct shape and device
            obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0)  # Add batch dimension
            
            # Move to same device as model if needed
            if hasattr(self.model.policy, 'device'):
                obs_tensor = obs_tensor.to(self.model.policy.device)
            
            # Get action probabilities from the model
            with torch.no_grad():
                action_probs = self.model.policy.get_distribution(obs_tensor).distribution.probs.squeeze().cpu().numpy()
            
            # Mask invalid actions by setting their probabilities to 0
            masked_probs = np.zeros_like(action_probs)
            for action in valid_actions:
                if action < len(masked_probs):
                    masked_probs[action] = action_probs[action]
            
            # Renormalize probabilities
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
                # Sample from the masked distribution
                action = np.random.choice(len(masked_probs), p=masked_probs)
            else:
                # Fallback: choose randomly from valid actions
                action = np.random.choice(valid_actions)
                
        except Exception as e:
            # Fallback to simple approach if anything fails
            print(f"Warning: Action masking failed ({e}), using fallback")
            action = np.random.choice(valid_actions)
        
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
    
    def _predict_simple(self, obs_array: np.ndarray, valid_actions: list) -> int:
        """
        Simple prediction with action masking - just uses model.predict() 
        and validates the result.
        
        Args:
            obs_array: Flattened observation
            valid_actions: List of valid action indices
            
        Returns:
            Valid action selected by the model
        """
        max_attempts = 10
        for attempt in range(max_attempts):
            action, _ = self.model.predict(obs_array, deterministic=False)
            action = int(action)
            
            if action in valid_actions:
                return action
            
            # If invalid action selected, try a few more times
            if attempt < max_attempts - 1:
                continue
            else:
                # Final fallback: choose randomly from valid actions
                import random
                return random.choice(valid_actions)
        
        # Should never reach here, but just in case
        import random
        return random.choice(valid_actions)
