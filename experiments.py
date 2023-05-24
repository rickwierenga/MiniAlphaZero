from typing import Any, Callable, Tuple
from tictactoe import TicTacToe
from connect4 import Connect4
from util import battle
from resnet import Network
from MCTS import run_mcts, Node, board2state
from minimax import minimax
import torch
from functools import partial
import os
import re
import numpy as np
import random

def alphazero_agent(game, net):
  root = Node(to_play=game.to_play, prior=0)
  action, root = run_mcts(game=game, root=root, net=net, device="cpu", temperature=0, explore=True)
  return action, root.get_value()

def policy_agent(game, net):
  encoded_boards = board2state(game)
  with torch.no_grad():
    policies, values = net(encoded_boards)
  actions_probs = torch.zeros(7)
  for action in game.get_legal_moves():
    actions_probs[action] = policies[0, action]
  action = torch.argmax(actions_probs)
  return action, values[action]

def MCTSvsMiniMax(agent=alphazero_agent):
    # Load trained MCTS model
    models = [file.path for file in os.scandir('data') if re.match(r".*connect4-model-[0-9]*.pt", file.path)]
    N_REPEATS = 50
    DEPTH = 4
    for model in models:
        net = Network(board_size=(6, 7), policy_shape=(7,), num_layers=10)
        net.load_state_dict(torch.load(model))
        # Agent that can be called with game and net and returns action and value
        agent_mcts = lambda game: agent(game, net)
        # Agent that can be called with game and returns action and value
        agent_minimax = lambda game: minimax(game, max_depth=DEPTH)
        # Play against minimax
        # Repeat some number of times to estimate performance
        n_wins_mcts = 0
        for _ in range(N_REPEATS):
            # MCTS plays first
            win, _ = battle(Connect4(), agent_mcts, agent_minimax, False)
            if win == 1:
                n_wins_mcts += 1
            # MCTS plays second
            win, _ = battle(Connect4(), agent_minimax, agent_mcts, False)
            if win == -1:
                n_wins_mcts += 1
        print(f"{agent.__name__} {model} wins from MiniMax depth={DEPTH} in {100*n_wins_mcts/(N_REPEATS*2)}% of games")
    
def benchmark():
    """Benchmark value found by alphazero by using benchmark dataset
    
    http://blog.gamesolver.org/solving-connect-four/02-test-protocol/
    """
    
    # Load trained MCTS model
    models = [file.path for file in os.scandir('data') if re.match(r".*connect4-model-[0-9]*.pt", file.path)]
    for model in models:
        net = Network(board_size=(6, 7), policy_shape=(7,), num_layers=10)
        net.load_state_dict(torch.load(model))
        # Read benchmark dataset
        tests = [file.path for file in os.scandir('data') if re.match(r".*Test_L[0-9]_R[0-9]", file.path)]
        scores = []
        states = []
        for test in tests:
            with open(test, 'r') as f:
                for line in f:
                    # Convert state to our game state notation
                    state, score = line.strip().split(" ")
                    game = Connect4()
                    for action in state:
                        game = game.next_state(int(action)-1)
                    state = board2state(game)
                    states.append(state)
                    scores.append(int(score))
        # Get estimated value of each state from MCTS
        with torch.no_grad():
            _, values = net(torch.stack(states))
            values = [value.item() for value in values]
        # Normalise benchmark scores: 0 is draw, -1 is loss and 1 is win
        scores = [np.sign(score) for score in scores]
        # for score, value in zip(scores, values):
        #     print(f"{score}, {value.item()}")
        # Compare difference
        rmse = np.sqrt(np.square(np.subtract(values, scores)).mean())
        print(model, rmse)

def calc_elo(score, prev_elo_1, prev_elo_2):
    """Find new elo ratings for two players based on previous and current score
    
    Score should be 0, 0.5 and 1 for loss, draw and win respectively from the perspective of player 1.

    Global constants:
    DEFAULT_ELO: Arbitrary initial value, will change based on data
    K_FACTOR: Sensitivity of rating change to score difference

    TODO adjust these constants based on data and more research
    """
    DEFAULT_ELO = 1000
    K_FACTOR = 32 
    # Set default elo values if not given
    if prev_elo_1 is None: prev_elo_1 = DEFAULT_ELO
    if prev_elo_2 is None: prev_elo_2 = DEFAULT_ELO
    # Find expected score based on ELO difference
    d = prev_elo_2 - prev_elo_1
    ratio = d / 400
    exp_ratio = (1 + 10**ratio)
    expected_score = 1 / exp_ratio
    # Find updated ELO
    elo_d = K_FACTOR * (score - expected_score)
    elo_1 = prev_elo_1 + elo_d
    elo_2 = prev_elo_2 - elo_d
    return elo_1, elo_2

def find_elo_values():
    """  Elo rating system for comparing two players 
    
    In this case the players are different iterations of the same model.
    """
    raise DeprecationWarning("Not used as we are not sure this is valid")
    with open("data/results.txt") as file:
        elo_old, elo_new = None, None
        for line in file:
            # Each iteration is seen as a new player
            # TODO keep previous elo?
            # Read results from file TODO how will input be given?
            columns = line.split(" ")
            old_wins, draws, new_wins = 0, 0, 0
            old_wins += int(columns[3])
            draws += int(columns[5])
            new_wins += int(columns[7])
            print(f"ELO ratings iteration {columns[1]}:")
            # print(f"- Player 'New' has won {new_wins} games")
            # print(f"- There have been {draws} draws")
            # print(f"- Player 'Old' has won {old_wins} games")
            # Generate scores as array of game results
            scores = [0 for _ in range(old_wins)] + [0.5 for _ in range(draws)] + [1 for _ in range(new_wins)]
            random.shuffle(scores) # TODO needed to shuffle?
            # Find elo values based on game results
            for i, score in enumerate(scores):
                elo_old, elo_new = calc_elo(score, elo_new, elo_old)
                # print(f"Game {i+1}: 'New' scored {score} against 'Old'")
                # print(f"Elo 'New': {elo_new}")
                # print(f"Elo 'Old': {elo_old}") 
            print(f"Final ELO ratings:")
            print(f"Elo 'New': {elo_new}")
            print(f"Elo 'Old': {elo_old}")
        # Reset elo for next iteration
        elo_old = elo_new # transitive, the new player is reused in the old
        elo_new = None

def performance_rating():
    raise NotImplementedError("May not be useful")

def find_performance_rating():
    raise NotImplementedError("May not be useful")

def test_sig_better():
    """Test if one model is significantly better than another"""
    with open("data/results.txt") as file:
        for line in file:
            # Parse results from file TODO which format?
            columns = line.split(" ")
            old_wins, draws, new_wins = 0, 0, 0
            old_wins += int(columns[3])
            draws += int(columns[5])
            new_wins += int(columns[7])
            print(f"Results iteration {columns[1]}:")
            # print(f"- Player 'New' has won {new_wins} games")
            # print(f"- There have been {draws} draws")
            # print(f"- Player 'Old' has won {old_wins} games")
            # Test if new player is significantly better than old player
            # Find 95% confidence interval
            z = 1.645 # 95% confidence interval, one-sided
            h0 = 0.5 # Null hypothesis: new player is not better than old player
            # Find average win rate from perspective of new player
            n = old_wins + draws + new_wins
            win_rate = new_wins / n # Proportion of games won by new player
            std = np.sqrt(h0 * (1 - h0) / n) # Standard error
            print(f"Win rate: {win_rate} ({std})")
            upper = h0 + z * std
            if win_rate > upper:
                print(f"{win_rate} is higher than {upper}")
                print("New player is significantly better than old player")
            else:
                print(f"{win_rate} is lower than {upper}")
                print("New player is not significantly better than old player")

def policyVsMiniMax():
    """Compare performance of policy network vs minimax"""
    MCTSvsMiniMax(agent=policy_agent)

if __name__ == "__main__":
    MCTSvsMiniMax()
