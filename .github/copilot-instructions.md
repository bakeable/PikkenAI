<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Pikken AI - Copilot Instructions

This project implements a reinforcement learning agent for the dice game Pikken using Python, Gymnasium, and Stable Baselines3.

## Project Context

- **Domain**: Reinforcement Learning, Game AI, Multi-agent Systems
- **Game**: Pikken - a dice game involving bluffing, bidding, and strategic decision-making
- **Framework**: Gymnasium environment with PPO training via Stable Baselines3
- **Architecture**: Modular agent system with self-play training capabilities

## Code Style Guidelines

- Follow PEP 8 Python style conventions
- Use type hints for function parameters and return values
- Write docstrings for all classes and functions
- Prefer composition over inheritance for agent behaviors
- Use meaningful variable names that reflect game concepts (e.g., `current_bid`, `players_alive`)

## Key Design Patterns

1. **Strategy Pattern**: Different agent types implement `BaseAgent.decide_action()`
2. **Environment Pattern**: `PikkenEnv` follows Gymnasium interface
3. **Observer Pattern**: Training callbacks monitor progress
4. **Factory Pattern**: Agent creation and environment setup

## Domain-Specific Terminology

- **Bid**: (quantity, face_value) tuple representing a claim about dice
- **Bluff**: Making an impossible or unlikely bid
- **Call Bluff**: Challenging the previous player's bid
- **Observation**: Game state visible to current player
- **Action Space**: 42 discrete actions (pass, call_bluff, raise_bid variants, bluff_bid variants)

## Common Tasks

When working on this project, you'll often need to:

1. **Add new agent types**: Inherit from `BaseAgent` and implement `decide_action()`
2. **Modify reward structure**: Edit `PikkenEnv.step()` method
3. **Adjust training parameters**: Modify `train.py` hyperparameters
4. **Add evaluation metrics**: Extend `Tournament` class in `evaluate.py`
5. **Create game variants**: Subclass `PikkenEnv` with rule modifications

## Testing and Validation

- Test agents with `evaluate.py` tournaments
- Validate environment with random agents first
- Use small timestep counts for debugging
- Monitor training with TensorBoard logs
- Check win rates are reasonable (not too high/low)

## Performance Considerations

- Vectorized environments for faster training
- Batch processing in evaluation
- Efficient observation space design
- Memory management for long training runs

## Dependencies and Imports

- Always check if optional dependencies are available (stable-baselines3, matplotlib)
- Use try/except blocks for imports
- Provide fallback behavior when dependencies missing
- Keep core game logic independent of ML libraries
