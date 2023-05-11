from typing import Any, Callable, Tuple
from tictactoe import TicTacToe
from connect4 import Connect4
from util import battle
from resnet import Network
from MCTS import run_mcts, Node
from minimax import minimax
import torch
from functools import partial
import os
import re

def alphazero_agent(game, net):
  root = Node(to_play=game.to_play, prior=0)
  action, root = run_mcts(game=game, root=root, net=net, num_searches=480, temperature=0, explore=True)
  return action, root.get_value()

def MCTSvsMiniMax():
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
    
if __name__ == "__main__":
    MCTSvsMiniMax()
