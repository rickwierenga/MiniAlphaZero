# basic MCTS implementation

import copy
import os
import pickle

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from game import Game
from tictactoe import TicTacToe

torch.random.manual_seed(0)
np.random.seed(0)

NUM_SIMULATIONS = 300
NUM_GAMES_SELF_PLAY = 150 # this will be removed once the self play is always running
NUM_ITERATIONS = 5 # number of iterations of self play, training, and evaluation
DEBUG = int(os.environ.get("DEBUG", 0))

C = 2


class Node:
  def __init__(self, to_play, state: Game, prior):
    self.to_play = to_play
    self.state = state
    self.children = {}
    self.prior = prior
    self.visit_count = 0
    self.value_sum = 0

  def add_child(self, action, prior):
    """ action: an action encoded as a tuple, the index into the array. """
    child_state = self.state.next_state(action=action, player=self.to_play)
    child = Node(to_play=-self.to_play, state=child_state, prior=prior)
    self.children[action] = child

  def get_value(self):
    if self.visit_count == 0: return 0
    return self.value_sum / self.visit_count

  def get_expanded(self):
    return len(self.children) > 0


class Network(nn.Module):
  def __init__(self):
    # input: 4x3x3 (3x3 board, 4 channels: player 1, player -1, empty, current player)
    # TODO: add a batch dimension

    super().__init__()

    self.network = nn.Sequential(
      nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(16 * 3 * 3, 64),
      nn.ReLU(),
      nn.Linear(64, 4 * 3 * 3),
      nn.ReLU(),
    )

    # output: scalar (value of the current state)
    self.value_head = nn.Sequential(
      nn.Linear(4 * 3 * 3, 16),
      nn.ReLU(),
      nn.Linear(16, 1),
      nn.Tanh()
    )

    # output: 3x3 (3x3 board, where should the next move be played)
    self.policy_head = nn.Sequential(
      nn.Linear(4 * 3 * 3, 64),
      nn.ReLU(),
      nn.Linear(64, 3 * 3),
      nn.Softmax(dim=1) # dim=1, why?
    )

  def board2state(self, board, current_player):
    # convert the board to a state that can be used by the network
    # the board is a 3x3 matrix with 1, -1, 0
    # we will convert this to a 3x3x3 matrix with 1s and 0s
    # the first channel will be 1 where the board has 1s, the second channel will be 1 where the board has -1s, the third channel will be 1 where the board has 0s the fourth channel will be the current player

    state = torch.zeros((4, 3, 3))
    board = torch.tensor(board)
    state[0, :, :] = board == 1
    state[1, :, :] = board == -1
    state[2, :, :] = board == 0
    state[3, :, :] = current_player

    return state

  def forward(self, encoded_board):
    # state = self.board2state(board, current_player)
    x = encoded_board
    x = x.unsqueeze(0) # add batch dimension
    x = self.network(x)
    x_value = self.value_head(x)
    x_policy = self.policy_head(x)
    x_policy = x_policy.reshape((3, 3)) # ?
    return x_policy, x_value


def uct_score(node: Node, parent: Node):
  # C = math.log((parent.visit_count + config.pb_c_base + 1) /
  #                 config.pb_c_base) + config.pb_c_init
  prior_score = C * node.prior * np.sqrt(parent.visit_count) / (1 + node.visit_count) # U
  return node.get_value() + prior_score

def expand(node: Node, net: Network):
  # obtain the policy and value from the network
  encoded_board = net.board2state(node.state.board, node.to_play)
  with torch.no_grad():
    policy, value = net(encoded_board)

  # for each available action, add a new state to the tree
  for action in node.state.get_legal_moves():
    node.add_child(action=action, prior=policy[action]) # action indexes into policy, which is a 3x3 matrix

  return value

def select_action(node):
  # TODO:
  # "For the first 30 moves of each game, the temperature is set to τ = 1; this selects moves
  # proportionally to their visit count in MCTS, and ensures a diverse set of positions are
  # encountered. For the remainder of the game, an infinitesimal temperature is used, τ→0."
  return max(node.children, key=lambda x: node.children[x].visit_count)

def run_mcts(root, net: Network, num_searches):
  # start by expanding the root node
  expand(node=root, net=net)

  # add Dirichlet noise to the root node
  epsilon = 0.25
  num_actions = len(root.children)
  noise = torch.distributions.dirichlet.Dirichlet(torch.ones((num_actions,)) * epsilon).sample()
  for i, node in enumerate(root.children.values()):
    node.prior = (1 - epsilon) * node.prior + epsilon * noise[i]

  for _ in range(num_searches):
    # select a leaf node according to UCT
    node, to_play = root, root.to_play
    path = [node]
    while node.get_expanded():
      node = max(node.children.values(), key=lambda x: uct_score(node=x, parent=node))
      path.append(node)

    # expand the leaf node
    to_play = node.to_play # perspective of the value
    value = expand(node=node, net=net)

    # backpropagate
    for node in path:
      node.value_sum += value if node.to_play == to_play else -value
      node.visit_count += 1

  # Return the action to play and the action probabilities (normalized visit counts).
  return select_action(root), root


def play_against_user(game, net):
  to_play = 1
  computer_player = 1

  root = Node(to_play=to_play, state=game, prior=0)

  while not game.is_terminal():
    print("\n")
    action, root = run_mcts(root=root, net=net, num_searches=300)

    game.print_board()
    print("available moves: ", game.get_legal_moves())

    if to_play == computer_player:
      # I think this prints the best move for the computer
      print("computer move: ", action, "value: ", root.get_value())
      game = game.next_state(action=action, player=to_play)
    else:
      action = game.get_user_action()

      print("computer says: best move", action, "value: ", root.get_value())

      game = game.next_state(action=action, player=to_play)

    if game.is_terminal():
      print("\n")
      print("game over")
      print("winner", game.get_winner())
      print("computer player", computer_player)
      print("\n")
      print("==== end ====")
      print("\n")
      break

    to_play = -to_play

    root = root.children[action]


buffer = []

import random
random.seed(0)
def self_play(net):
  global buffer

  for game_num in range(NUM_GAMES_SELF_PLAY):
    game = TicTacToe(board=np.zeros((3, 3)))
    to_play = 1

    root = Node(state=game, to_play=to_play, prior=0)

    game_buffer = []
    while not game.is_terminal():
      state = game.board
      action, root = run_mcts(root=root, net=net, num_searches=NUM_SIMULATIONS)

      mcts_policy = torch.zeros((3, 3))
      for a, n in root.children.items():
        mcts_policy[a] = n.visit_count
      mcts_policy = mcts_policy / mcts_policy.sum() # renormalize

      game_buffer.append(((state, to_play), mcts_policy))

      game = game.next_state(action=action, player=to_play)
      to_play = -to_play

      # reuse the tree
      root = root.children[action]

    winner = game.get_winner()
    replay_player = 1 # start with the first player, like in the game
    for state, probabilities in game_buffer:
      buffer.append((state, probabilities, winner * replay_player)) # trick: multiply by the winner to flip the sign if the winner is -1
      replay_player = -replay_player

    print('  self play game', game_num, "winner", winner)

def train(net, optim):
  global buffer

  print("  - training -")

  for batch in buffer: # TODO: actually sample and use batches
    state, probabilities, value = batch
    board, current_player = state
    value = torch.tensor([[value]], dtype=torch.float32) # add batch dimension

    net.zero_grad() # zeroes the gradient buffers of all parameters. is that necessary?

    encoded_board = net.board2state(board, current_player)
    predicted_policy, predicted_value = net(encoded_board)

    assert predicted_policy.shape == probabilities.shape
    assert predicted_value.shape == value.shape
    loss = F.cross_entropy(predicted_policy, probabilities) + F.mse_loss(predicted_value, value)
    loss.backward()

    optim.step()

  # only keep the last 2 * NUM_GAMES_SELF_PLAY games
  if len(buffer) > 2 * NUM_GAMES_SELF_PLAY:
    buffer = buffer[-2 * NUM_GAMES_SELF_PLAY:]

def battle(old_net, new_net):
  NUM_EVALUATE_GAMES = 50
  old_wins = 0
  draws = 0
  new_wins = 0

  for game_num in range(NUM_EVALUATE_GAMES):
    game = TicTacToe(board=np.zeros((3, 3)))
    to_play = 1
    old_net_is_player1 = game_num % 2 == 0

    # TODO: reuse the tree for each net.
    while not game.is_terminal():
      root = Node(state=game, to_play=to_play, prior=0)
      use_old_net = (to_play == 1 and old_net_is_player1) or (to_play == -1 and not old_net_is_player1)
      action, root = run_mcts(root=root, net=old_net if use_old_net else new_net, num_searches=300)
      mcts_policy = torch.zeros((3, 3)) # policy-like
      for a, n in root.children.items():
        mcts_policy[a] = n.visit_count
      mcts_policy = mcts_policy / mcts_policy.sum() # renormalize
      game = game.next_state(action=action, player=to_play)
      to_play = -to_play

    winner = game.get_winner()
    if winner == 0:
      draws += 1
    elif winner == 1:
      if old_net_is_player1:
        old_wins += 1
      else:
        new_wins += 1
    else:
      if old_net_is_player1:
        new_wins += 1
      else:
        old_wins += 1

    print('  eval play game', game_num, "winner", winner, "old player won", old_net_is_player1 and winner == 1 or not old_net_is_player1 and winner == -1)

  return old_wins, draws, new_wins

def demo_play(net):
  # play network against network
  game = TicTacToe()

  to_play = 1

  root = Node(state=game, to_play=to_play, prior=0)
  while not game.is_terminal():
    action_to_play, _ = run_mcts(root=root, net=net, num_searches=300)
    game.print_board()
    print()
    game = game.next_state(action=action_to_play, player=to_play)
    to_play = -to_play
    root = root.children[action_to_play]

  print("winner", game.get_winner())
  print(game.board)

if __name__ == "__main__":
  net = Network()
  optim = torch.optim.Adam(net.parameters(), lr=0.01) # 0.001

  # load model and optimizer
  net.load_state_dict(torch.load('model.pt'))
  optim.load_state_dict(torch.load('optimizer.pt'))

  for iteration in range(NUM_ITERATIONS):
    # copy the network
    net_copy = copy.deepcopy(net)
    optim_copy = copy.deepcopy(optim)

    print("iteration", iteration)
    self_play(net)
    train(net, optim)

    # play against previous version
    old_wins, draws, new_wins = battle(net_copy, net)
    print("old wins", old_wins, "draws", draws, "new wins", new_wins)
    if old_wins > new_wins:
      print(" >>> choose OLD model !!!")
      net = net_copy
      optim = optim_copy
    else:
      print(" >>> choose NEW model !!!")

      # save model
      torch.save(net.state_dict(), 'model.pt')
      # save optimizer
      torch.save(optim.state_dict(), 'optimizer.pt')
