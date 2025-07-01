"""
Evaluation script for Pikken AI agents.

This script runs tournaments between different agents and provides
detailed statistics about their performance.
"""

import argparse
import time
import numpy as np
from typing import List, Dict, Tuple
import os

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from pikken_env import PikkenEnv
from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.rl_agent import RLAgent


class Tournament:
    """
    Tournament manager for evaluating agents against each other.
    """
    
    def __init__(self, agents: List[BaseAgent], num_games: int = 100):
        self.agents = agents
        self.num_games = num_games
        self.results = {}
        self.game_logs = []
    
    def run_tournament(self, verbose: bool = True) -> Dict:
        """
        Run a round-robin tournament between all agents.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing tournament results
        """
        if verbose:
            print(f"Running tournament with {len(self.agents)} agents...")
            print(f"Games per matchup: {self.num_games}")
        
        # Reset agent statistics
        for agent in self.agents:
            agent.reset_stats()
        
        total_games = 0
        start_time = time.time()
        
        # Run games with all possible agent combinations
        for game_idx in range(self.num_games):
            if verbose and game_idx % 20 == 0:
                print(f"Game {game_idx + 1}/{self.num_games}")
            
            # Randomly shuffle agents for this game
            game_agents = np.random.choice(self.agents, size=4, replace=True).tolist()
            winner_idx = self._play_game(game_agents)
            
            # Update statistics
            if winner_idx is not None:
                game_agents[winner_idx].update_stats(True)
                for i, agent in enumerate(game_agents):
                    if i != winner_idx:
                        agent.update_stats(False)
            
            total_games += 1
        
        elapsed_time = time.time() - start_time
        
        # Compile results
        results = {
            'tournament_info': {
                'num_agents': len(self.agents),
                'games_played': total_games,
                'duration_seconds': elapsed_time
            },
            'agent_stats': []
        }
        
        for agent in self.agents:
            stats = {
                'name': agent.name,
                'games_played': agent.games_played,
                'games_won': agent.games_won,
                'win_rate': agent.get_win_rate()
            }
            results['agent_stats'].append(stats)
        
        if verbose:
            self._print_results(results)
        
        return results
    
    def _play_game(self, agents: List[BaseAgent]) -> int:
        """
        Play a single game between the given agents.
        
        Returns:
            Index of winning agent, or None if error
        """
        env = PikkenEnv(num_players=len(agents))
        observation, info = env.reset()
        
        max_steps = 1000  # Prevent infinite games
        step_count = 0
        
        while step_count < max_steps:
            current_player = info['current_player']
            
            # Skip if player is dead
            if not info['players_alive'][current_player]:
                action = 0  # pass
            else:
                # Get action from current agent
                try:
                    action = agents[current_player].decide_action(observation, info)
                except Exception as e:
                    print(f"Error getting action from {agents[current_player].name}: {e}")
                    action = 0  # default to pass
            
            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                # Find winner (last player alive)
                alive_players = [i for i, alive in enumerate(info['players_alive']) if alive]
                if len(alive_players) == 1:
                    return alive_players[0]
                elif len(alive_players) == 0:
                    return None  # Draw/error
                else:
                    # Multiple players alive at termination, pick randomly
                    return np.random.choice(alive_players)
            
            step_count += 1
        
        # Game too long, pick random winner from alive players
        alive_players = [i for i, alive in enumerate(info['players_alive']) if alive]
        if alive_players:
            return np.random.choice(alive_players)
        return None
    
    def _print_results(self, results: Dict):
        """Print tournament results in a formatted table."""
        print("\n" + "="*60)
        print("TOURNAMENT RESULTS")
        print("="*60)
        
        # Sort by win rate
        sorted_agents = sorted(
            results['agent_stats'],
            key=lambda x: x['win_rate'],
            reverse=True
        )
        
        print(f"{'Rank':<6} {'Agent':<20} {'Games':<8} {'Wins':<8} {'Win Rate':<10}")
        print("-" * 60)
        
        for i, stats in enumerate(sorted_agents):
            rank = i + 1
            name = stats['name']
            games = stats['games_played']
            wins = stats['games_won']
            win_rate = f"{stats['win_rate']:.1%}"
            
            print(f"{rank:<6} {name:<20} {games:<8} {wins:<8} {win_rate:<10}")
        
        info = results['tournament_info']
        print(f"\nTotal games: {info['games_played']}")
        print(f"Duration: {info['duration_seconds']:.1f} seconds")


def create_agents(rl_model_path: str = None) -> List[BaseAgent]:
    """Create a list of agents for evaluation."""
    agents = [
        RandomAgent("Random-1"),
        RandomAgent("Random-2"),
        HeuristicAgent("Conservative", aggressiveness=0.2),
        HeuristicAgent("Aggressive", aggressiveness=0.8),
        HeuristicAgent("Balanced", aggressiveness=0.5),
    ]
    
    # Add RL agent if model exists
    if rl_model_path and os.path.exists(rl_model_path + ".zip"):
        try:
            rl_agent = RLAgent("RL-Agent", model_path=rl_model_path)
            agents.append(rl_agent)
            print(f"Loaded RL agent from {rl_model_path}")
        except Exception as e:
            print(f"Could not load RL agent: {e}")
    else:
        print("No RL model found, using rule-based agents only")
    
    return agents


def plot_results(results: Dict, save_path: str = "tournament_results.png"):
    """Plot tournament results."""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    agent_stats = results['agent_stats']
    names = [stats['name'] for stats in agent_stats]
    win_rates = [stats['win_rate'] for stats in agent_stats]
    
    # Sort by win rate
    sorted_indices = np.argsort(win_rates)[::-1]
    names = [names[i] for i in sorted_indices]
    win_rates = [win_rates[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, win_rates)
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.title('Pikken AI Tournament Results')
    plt.xlabel('Agent')
    plt.ylabel('Win Rate')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (name, rate) in enumerate(zip(names, win_rates)):
        plt.text(i, rate + 0.01, f'{rate:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Results plot saved to {save_path}")


def head_to_head_evaluation(agent1: BaseAgent, agent2: BaseAgent, num_games: int = 100):
    """Run head-to-head evaluation between two agents."""
    print(f"\nHead-to-head: {agent1.name} vs {agent2.name}")
    print(f"Games: {num_games}")
    
    agent1.reset_stats()
    agent2.reset_stats()
    
    for game_idx in range(num_games):
        # Alternate who goes first
        if game_idx % 2 == 0:
            agents = [agent1, agent2, RandomAgent(), RandomAgent()]
        else:
            agents = [agent2, agent1, RandomAgent(), RandomAgent()]
        
        env = PikkenEnv(num_players=4)
        observation, info = env.reset()
        
        max_steps = 500
        step_count = 0
        
        while step_count < max_steps:
            current_player = info['current_player']
            
            if not info['players_alive'][current_player]:
                action = 0
            else:
                action = agents[current_player].decide_action(observation, info)
            
            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                alive_players = [i for i, alive in enumerate(info['players_alive']) if alive]
                if len(alive_players) == 1:
                    winner_idx = alive_players[0]
                    if winner_idx < 2:  # One of our test agents won
                        agents[winner_idx].update_stats(True)
                        agents[1-winner_idx].update_stats(False)
                break
            
            step_count += 1
    
    print(f"{agent1.name}: {agent1.games_won}/{agent1.games_played} ({agent1.get_win_rate():.1%})")
    print(f"{agent2.name}: {agent2.games_won}/{agent2.games_played} ({agent2.get_win_rate():.1%})")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Pikken AI agents")
    parser.add_argument(
        "--model-path", type=str, default="./models/pikken_agent",
        help="Path to trained RL model"
    )
    parser.add_argument(
        "--games", type=int, default=100,
        help="Number of games per tournament"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Plot results"
    )
    parser.add_argument(
        "--head-to-head", action="store_true",
        help="Run head-to-head evaluation"
    )
    
    args = parser.parse_args()
    
    print("=== Pikken AI Evaluation ===")
    
    # Create agents
    agents = create_agents(args.model_path)
    print(f"Created {len(agents)} agents: {[agent.name for agent in agents]}")
    
    # Run tournament
    tournament = Tournament(agents, num_games=args.games)
    results = tournament.run_tournament()
    
    if args.plot:
        plot_results(results)
    
    if args.head_to_head and len(agents) >= 2:
        # Run head-to-head between top agents
        sorted_agents = sorted(agents, key=lambda x: x.get_win_rate(), reverse=True)
        head_to_head_evaluation(sorted_agents[0], sorted_agents[1], num_games=50)


if __name__ == "__main__":
    main()
