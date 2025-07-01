#!/usr/bin/env python3
"""
Evaluation script for the Pikken RL agent.
Tests the trained agent against various opponents in tournaments.
"""

import argparse
import os
import numpy as np
from typing import List, Dict, Tuple
from pikken_env import PikkenEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.rl_agent import RLAgent

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class Tournament:
    """Manages tournaments between different agent types."""
    
    def __init__(self, num_players: int = 5):
        self.num_players = num_players
        self.results = {}
        
    def run_tournament(self, agents: List, num_games: int = 100, verbose: bool = True) -> Dict:
        """Run a tournament between the given agents."""
        if len(agents) != self.num_players:
            raise ValueError(f"Expected {self.num_players} agents, got {len(agents)}")
        
        wins = {i: 0 for i in range(self.num_players)}
        game_lengths = []
        invalid_action_counts = {i: 0 for i in range(self.num_players)}
        
        if verbose:
            print(f"🏆 Running tournament with {num_games} games")
            for i, agent in enumerate(agents):
                print(f"   Player {i}: {agent.name}")
        
        for game_num in range(num_games):
            if verbose and (game_num + 1) % (num_games // 10) == 0:
                print(f"   Progress: {game_num + 1}/{num_games} games")
            
            winner, steps, invalid_actions = self._run_single_game(agents)
            
            if winner is not None:
                wins[winner] += 1
                game_lengths.append(steps)
                for player, count in invalid_actions.items():
                    invalid_action_counts[player] += count
        
        # Calculate results
        results = {
            'wins': wins,
            'win_rates': {i: wins[i] / num_games for i in range(self.num_players)},
            'avg_game_length': np.mean(game_lengths) if game_lengths else 0,
            'invalid_actions': invalid_action_counts,
            'total_games': num_games,
            'completed_games': len(game_lengths)
        }
        
        self.results = results
        return results
    
    def _run_single_game(self, agents: List) -> Tuple[int, int, Dict[int, int]]:
        """Run a single game and return winner, steps, and invalid action counts."""
        env = PikkenEnv(num_players=self.num_players)
        observation, info = env.reset()
        
        steps = 0
        max_steps = 20000
        invalid_actions = {i: 0 for i in range(self.num_players)}
        
        while steps < max_steps:
            current_player = env.current_player
            current_agent = agents[current_player]
            
            # Get observation and info for current player
            current_obs = observation.get(current_player, observation)
            current_info = info.get(current_player, info)
            
            try:
                action = current_agent.decide_action(current_obs, current_info)
                observation, rewards, terminated, info = env.step(action)
                
                # Track invalid actions if penalties are given
                if isinstance(rewards, dict) and current_player in rewards:
                    if rewards[current_player] < -0.1:  # Penalty threshold
                        invalid_actions[current_player] += 1
                elif isinstance(rewards, (int, float)) and rewards < -0.1:
                    invalid_actions[current_player] += 1
                
                # Check for game end - first player to reach 0 dice wins
                winner = None
                for player_idx in range(self.num_players):
                    if len(env.players_dice[player_idx]) == 0:
                        winner = player_idx
                        break
                
                if winner is not None:
                    return winner, steps, invalid_actions
                
                if any(terminated.values() if isinstance(terminated, dict) else [terminated]):
                    # Game ended without clear winner - shouldn't happen in Pikken
                    return None, steps, invalid_actions
                    
            except Exception as e:
                print(f"   ❌ Error in game: {e}")
                return None, steps, invalid_actions
            
            steps += 1
        
        # Game exceeded max steps
        return None, steps, invalid_actions
    
    def print_results(self):
        """Print tournament results in a readable format."""
        if not self.results:
            print("No results to display")
            return
        
        results = self.results
        print(f"\n📊 Tournament Results ({results['completed_games']}/{results['total_games']} games completed)")
        print(f"   Average game length: {results['avg_game_length']:.1f} steps")
        print("\n🏆 Win Rates:")
        
        sorted_players = sorted(range(self.num_players), 
                              key=lambda i: results['win_rates'][i], reverse=True)
        
        for rank, player in enumerate(sorted_players, 1):
            wins = results['wins'][player]
            win_rate = results['win_rates'][player] * 100
            invalid = results['invalid_actions'][player]
            print(f"   {rank}. Player {player}: {wins} wins ({win_rate:.1f}%) - {invalid} invalid actions")
    
    def plot_results(self, save_path: str = None):
        """Plot tournament results."""
        if not MATPLOTLIB_AVAILABLE:
            print("❌ Matplotlib not available for plotting")
            return
        
        if not self.results:
            print("No results to plot")
            return
        
        results = self.results
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Win rates plot
        players = list(range(self.num_players))
        win_rates = [results['win_rates'][i] * 100 for i in players]
        
        bars1 = ax1.bar(players, win_rates, color=['red', 'blue', 'green', 'orange', 'purple'][:self.num_players])
        ax1.set_xlabel('Player')
        ax1.set_ylabel('Win Rate (%)')
        ax1.set_title('Win Rates by Player')
        ax1.set_ylim(0, max(win_rates) * 1.1 if win_rates else 100)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, win_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # Invalid actions plot
        invalid_counts = [results['invalid_actions'][i] for i in players]
        
        bars2 = ax2.bar(players, invalid_counts, color=['red', 'blue', 'green', 'orange', 'purple'][:self.num_players])
        ax2.set_xlabel('Player')
        ax2.set_ylabel('Invalid Actions')
        ax2.set_title('Invalid Actions by Player')
        
        # Add value labels on bars
        for bar, count in zip(bars2, invalid_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        else:
            plt.show()

def create_agent_lineup(rl_model_path: str = None) -> List:
    """Create a lineup of agents for testing."""
    agents = []
    
    # Add RL agents if model exists
    if rl_model_path and os.path.exists(rl_model_path):
        try:
            agents.append(RLAgent("TrainedRL", model_path=rl_model_path))
            print("✅ RL agent loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load RL agent: {e}")
            agents.append(RandomAgent("RL_Fallback"))
    else:
        if rl_model_path:
            print(f"❌ Model file not found: {rl_model_path}")
        agents.append(RandomAgent("NoModel"))
    
    # Add diverse opponent agents
    agents.extend([
        RandomAgent("Random"),
        HeuristicAgent("Conservative", aggressiveness=0.2),
        HeuristicAgent("Moderate", aggressiveness=0.5),
        HeuristicAgent("Aggressive", aggressiveness=0.8)
    ])
    
    return agents

def main():
    """Main evaluation function with command line interface."""
    parser = argparse.ArgumentParser(description="Evaluate Pikken RL agents")
    parser.add_argument("--model-path", type=str, default="./models/pikken_agent.zip",
                       help="Path to the trained RL model")
    parser.add_argument("--games", type=int, default=100,
                       help="Number of games to play")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots of results")
    parser.add_argument("--plot-save", type=str,
                       help="Save plot to file instead of displaying")
    parser.add_argument("--players", type=int, default=5,
                       help="Number of players (default: 5)")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    print("🎲 Pikken AI Evaluation")
    print("=" * 50)
    
    # Create agents
    agents = create_agent_lineup(args.model_path)
    
    if len(agents) != args.players:
        print(f"❌ Expected {args.players} agents, but got {len(agents)}")
        return
    
    # Run tournament
    tournament = Tournament(num_players=args.players)
    results = tournament.run_tournament(agents, args.games, verbose=not args.quiet)
    
    # Display results
    tournament.print_results()
    
    # Generate plots if requested
    if args.plot or args.plot_save:
        tournament.plot_results(args.plot_save)
    
    # Save results
    os.makedirs("eval_logs", exist_ok=True)
    np.savez("eval_logs/evaluations.npz", **results)
    print(f"\n💾 Results saved to eval_logs/evaluations.npz")

if __name__ == "__main__":
    main()
