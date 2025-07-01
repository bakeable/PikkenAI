# Pikken AI - Self-learning Neural Bluffer

A reinforcement learning agent that learns to play the dice game Pikken through self-play and strategic bluffing.

## Overview

Pikken AI implements a complete framework for training AI agents to play Pikken, a dice game involving bluffing, bidding, and risk assessment. The project uses modern reinforcement learning techniques with Stable Baselines3 and provides a gym-compatible environment for training and evaluation.

## Game Rules

🎯 **Objective**

Be the first player to get rid of all your dice — by bidding wisely, bluffing boldly, or calling "Bullshit!" at just the right moment.

---

🎲 **Setup**

- Each player starts with **5 dice**.
- All players roll their dice in secret.
- The **Pik** (the one) is a wild die and counts **double** towards any declared value.

---

🗣️ **Bidding**

- The starting player makes a bid, e.g. `"5 threes"` — meaning they claim that at least **5 dice** among all players show a **3 or a Pik**.
- The next player must:
  - Make a **higher bid** (by increasing the quantity or the face value), or
  - Call "**Bullshit!**" to challenge the bid.

---

🤥 **Calling Bullshit**

- All players must choose a side: either support the **bidder** or the **challenger**.
- Once sides are chosen, all dice are revealed.
- The bid is verified:
  - ✅ If the bid **is valid**, the **bidder and their supporters** each discard 1 die.
  - ❌ If the bid **is invalid**, the **challenger and their supporters** each discard 1 die.
- Losing a die brings a player closer to **winning** (by running out first).

---

👥 **Supporters** (Optional Rule)

- Neutral players can support either side in a challenge.
- If their side wins, they also discard a die.

---

🔁 **Next Round**

- All remaining players re-roll their dice.
- The **challenger** from the last round starts the new bid. If eliminated, the next player clockwise begins.
- Repeat until **one player has no dice left** — they win the game.

## Project Structure

```
PikkenAI/
├── pikken_env.py          # Gym-compatible game environment
├── agents/                # Different agent implementations
│   ├── __init__.py
│   ├── base_agent.py      # Abstract base agent class
│   ├── random_agent.py    # Random action agent
│   ├── heuristic_agent.py # Rule-based strategic agent
│   └── rl_agent.py        # Reinforcement learning agent
├── train.py               # Training script for RL agents
├── evaluate.py            # Tournament and evaluation system
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd PikkenAI
```

2. Create a virtual environment:

```bash
python -m venv pikken_env
source pikken_env/bin/activate  # On Windows: pikken_env\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

1. **Test the Environment**:

```python
from pikken_env import PikkenEnv
from agents.random_agent import RandomAgent

env = PikkenEnv(num_players=4)
agent = RandomAgent()

observation, info = env.reset()
action = agent.decide_action(observation, info)
observation, reward, terminated, truncated, info = env.step(action)
```

2. **Run a Tournament**:

```bash
python evaluate.py --games 100 --plot
```

3. **Train an RL Agent**:

```bash
python train.py --timesteps 100000 --save-path ./models/my_agent
```

4. **Evaluate Trained Agent**:

```bash
python evaluate.py --model-path ./models/my_agent --games 200 --plot
```

## Components

### Environment (`pikken_env.py`)

The `PikkenEnv` class implements a Gymnasium-compatible environment with:

- **Action Space**: 42 discrete actions (pass, call_bluff, raise_bid, bluff_bid)
- **Observation Space**: 15-dimensional vector containing own dice, current bid, game state
- **Reward Structure**: Designed to encourage winning while penalizing invalid moves

### Agents

#### Random Agent (`agents/random_agent.py`)

- Makes random valid moves
- Useful as baseline for evaluation

#### Heuristic Agent (`agents/heuristic_agent.py`)

- Uses probability calculations and game logic
- Configurable aggressiveness parameter
- Considers own dice when making decisions

#### RL Agent (`agents/rl_agent.py`)

- Uses PPO algorithm from Stable Baselines3
- Trainable through self-play
- Can be saved and loaded

### Training (`train.py`)

Features include:

- Self-play training against multiple opponent types
- Evaluation callbacks and checkpointing
- TensorBoard logging
- Configurable hyperparameters

### Evaluation (`evaluate.py`)

Comprehensive evaluation system with:

- Round-robin tournaments
- Head-to-head comparisons
- Statistical analysis and plotting
- Win rate tracking

## Advanced Usage

### Custom Agents

Create custom agents by inheriting from `BaseAgent`:

```python
from agents.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def decide_action(self, observation, info):
        # Your custom logic here
        return action
```

### Training Configuration

Customize training parameters:

```python
from train import train_agent

model = train_agent(
    total_timesteps=500_000,
    save_path="./models/expert_agent",
    eval_freq=5_000
)
```

### Environment Customization

Modify game rules by subclassing `PikkenEnv`:

```python
class CustomPikkenEnv(PikkenEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom modifications
```

## Performance Benchmarks

Typical performance on a modern CPU:

- Training: ~1000 steps/second
- Evaluation: ~100 games/second
- Model size: ~50KB for trained PPO agent

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir ./tensorboard_logs/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes linting (black, flake8)
5. Submit a pull request

## Dependencies

- `gymnasium`: Environment interface
- `stable-baselines3`: Reinforcement learning algorithms
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `torch`: Neural network backend

## License

MIT License - see LICENSE file for details.

## Future Improvements

- [ ] Add more sophisticated neural network architectures
- [ ] Implement curriculum learning
- [ ] Add multi-agent RL algorithms (MADDPG, etc.)
- [ ] Create web interface for human vs AI play
- [ ] Add more game variants and rule sets
- [ ] Implement explainable AI features

## Citation

If you use this project in research, please cite:

```bibtex
@software{pikken_ai,
  title={Pikken AI: Self-learning Neural Bluffer},
  author={Robin Bakker},
  year={2025},
  url={https://github.com/username/PikkenAI}
}
```
