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

def alphazero_agent(game, net, num_searches=None):
  root = Node(to_play=game.to_play, prior=0)
  action, root = run_mcts(game=game, root=root, net=net, num_searches=num_searches, temperature=0, explore=True)
  return action, root.get_value()

def MCTSvsMiniMax(num_searches=None):
    # Load trained MCTS model
    models = [file for file in os.listdir() if re.match(r"connect4-model-[0-9]*.pt", file)]
    N_REPEATS = 5
    DEPTH = 4
    for model in models:
        net = Network(board_size=(6, 7), policy_shape=(7,), num_layers=10)
        net.load_state_dict(torch.load(model))
        # Agent that can be called with game and net and returns action and value
        agent_mcts = lambda game: alphazero_agent(game, net)
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
        print(f"MCTS {model} wins from MiniMax depth={DEPTH} in {100*n_wins_mcts/(N_REPEATS*2)}% of games")
    
def benchmark():
    """Benchmark value found by alphazero by using benchmark dataset
    
    http://blog.gamesolver.org/solving-connect-four/02-test-protocol/
    """
    
    # Load trained MCTS model
    model = "data/connect4-model.pt"
    net = Network(board_size=(6, 7), policy_shape=(7,), num_layers=10)
    net.load_state_dict(torch.load(model))
    # Read benchmark dataset
    tests = [file for file in os.listdir() if re.match(r"Test_*", file)]
    scores = []
    states = []
    for test in tests:
        with open(test) as f:
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
    mse = np.square(np.subtract(values, scores)).mean()
    print(mse)

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
    pass

def find_performance_rating():
    pass

def policyVsMiniMax():
    """Compare performance of policy network vs minimax"""
    MCTSvsMiniMax(num_searches=0)

if __name__ == "__main__":
    find_elo_values()
