# basic MCTS implementation

import copy
import multiprocessing
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from game import Game
from tictactoe import TicTacToe

NUM_SEARCHES = 300
NUM_GAMES_SELF_PLAY = 50 # this will be removed once the self play is always running
NUM_ITERATIONS = 10 # number of iterations of self play, training, and evaluation
NUM_SELF_PLAYERS = 8

NUM_SAMPLING_MOVES = 2

C = 1.41

DEBUG = os.environ.get("DEBUG", 0)
if DEBUG:
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)

  NUM_GAMES_SELF_PLAY = 1
  NUM_SELF_PLAYERS = 1
  NUM_ITERATIONS = 1


class Node:
  def __init__(self, to_play, prior):
    self.to_play = to_play
    self.children = {}
    self.prior = prior
    self.visit_count = 0
    self.value_sum = 0

  def add_child(self, action, prior):
    """ action: an action encoded as a tuple, the index into the array. """
    child = Node(to_play=-self.to_play, prior=prior)
    self.children[action] = child

  def get_value(self):
    if self.visit_count == 0: return 0
    return self.value_sum / self.visit_count

  def get_expanded(self):
    return len(self.children) > 0

  def __repr__(self): return f"Node(value={self.get_value()}, prior={self.prior:.2f}, visit_count={self.visit_count}, to_play={self.to_play})"
  def visualize(self, level=0):
    if level == 0: print(self)
    for action, node in self.children.items():
      print("  " * level, action, node)
      node.visualize(level=level + 1)


class Network(nn.Module):
  def __init__(self):
    # input: 4x3x3 (3x3 board, 4 channels: player 1, player -1, empty, current player)
    # TODO: add a batch dimension

    super().__init__()

    self.network = nn.Sequential(
      nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(16 * 3 * 3, 8 * 3 * 3),
      nn.ReLU(),
      nn.Linear(8 * 3 * 3, 6 * 3 * 3),
      nn.ReLU(),
      nn.Linear(6 * 3 * 3, 4 * 3 * 3),
      nn.ReLU(),
    )

    # output: scalar (value of the current state)
    self.value_head = nn.Sequential(
      nn.Linear(4 * 3 * 3, 20),
      nn.ReLU(),
      nn.Linear(20, 1),
      nn.Tanh()
    )

    # output: 3x3 (3x3 board, where should the next move be played)
    self.policy_head = nn.Sequential(
      nn.Linear(4 * 3 * 3, 25),
      nn.ReLU(),
      nn.Linear(25, 3 * 3),
      nn.Softmax(dim=1) # dim=1, why?
    )

  def forward(self, encoded_board):
    x = encoded_board
    x = x.unsqueeze(0) # add batch dimension
    x = self.network(x)
    x_value = self.value_head(x)
    x_policy = self.policy_head(x)
    x_policy = x_policy.reshape((3, 3)) # ?
    return x_policy, x_value

def board2state(game):
  state = torch.zeros((4, 3, 3))
  board = torch.tensor(game.board)
  state[0, :, :] = board == 1
  state[1, :, :] = board == -1
  state[2, :, :] = board == 0
  state[3, :, :] = game.to_play
  return state


def uct_score(node: Node, parent: Node):
  # C = math.log((parent.visit_count + config.pb_c_base + 1) /
  #                 config.pb_c_base) + config.pb_c_init
  prior_score = C * node.prior * np.sqrt(parent.visit_count) / (1 + node.visit_count) # U
  return node.get_value() + prior_score

def expand(node: Node, game: Game, net: Network):
  # obtain the policy and value from the network
  encoded_board = board2state(game)
  with torch.no_grad():
    policy, value = net(encoded_board)

  # for each available action, add a new state to the tree
  for action in game.get_legal_moves():
    node.add_child(action=action, prior=policy[action]) # action indexes into policy, which is a 3x3 matrix

  return value

def select_action(node: Node, temperature: float):
  if temperature == 0: # strongest move: the one with the highest visit count
    return max(node.children, key=lambda x: node.children[x].visit_count)

  # select an action according to the visit counts
  visit_counts = np.array([child.visit_count for child in node.children.values()])
  visit_counts = visit_counts ** (1 / temperature)
  visit_counts = visit_counts / visit_counts.sum()
  # hacky, but numpy does not like us selecting from a list of tuples (thinks it's 2d)
  action_index = np.random.choice(len(visit_counts), p=visit_counts)
  action = list(node.children.keys())[action_index]
  return action

def run_mcts(game: Game, net: Network, num_searches: int, temperature: float, root=None, explore: bool = True):
  # start by expanding the root node
  if root is None:
    root = Node(to_play=game.to_play, prior=0)
    expand(node=root, net=net, game=game)

  # add Dirichlet noise to the root node
  epsilon = 0.4
  num_actions = len(root.children)
  noise = torch.distributions.dirichlet.Dirichlet(torch.ones((num_actions,)) * epsilon).sample()
  for i, node in enumerate(root.children.values()):
    node.prior = (1 - epsilon) * node.prior + epsilon * noise[i]

  for _ in range(num_searches):
    # select a leaf node according to UCT
    node = root
    path = [node]
    scratch_game = copy.deepcopy(game)
    while node.get_expanded():
      action, node = max(node.children.items(), key=lambda x: uct_score(node=x[1], parent=node))
      path.append(node)
      scratch_game = scratch_game.next_state(action=action) 

    # expand the leaf node
    value = expand(node=node, net=net, game=scratch_game)
    # this is the value from the perspective of who's to play in the leaf node.

    # backpropagate
    for node in path:
      node.value_sum += value if node.to_play == scratch_game.to_play else -value
      node.visit_count += 1

  # Return the action to play and the action probabilities (normalized visit counts).
  temperature = 1 if len(game.history) < NUM_SAMPLING_MOVES and explore else 0
  action = select_action(root, temperature=temperature), root
  if DEBUG:
    root.visualize()
    print("choosing", action)
  return action


buffer = []

def self_play(net):
  print("  - self play -")

  self_play_buffer = []

  for game_num in range(NUM_GAMES_SELF_PLAY):
    t = time.monotonic_ns()
    game = TicTacToe()
    root = Node(to_play=1, prior=0)
    game_buffer = []

    while not game.is_terminal():
      action, root = run_mcts(root=root, game=game, net=net, num_searches=NUM_SEARCHES, temperature=1, explore=True)

      mcts_policy = torch.zeros((3, 3))
      for a, n in root.children.items():
        mcts_policy[a] = n.visit_count
      mcts_policy = mcts_policy / mcts_policy.sum() # renormalize

      game_buffer.append((game, mcts_policy))

      game = game.next_state(action=action)
      # game.print_board()

      root = root.children[action] # reuse the tree

    winner = game.get_winner()
    for game, probabilities in game_buffer:
      self_play_buffer.append((game, probabilities, winner * game.to_play)) # trick: multiply by the winner to flip the sign if the winner is -1

    print(f'  self play game {game_num:03} winner {winner:02} took {(time.monotonic_ns() - t) / 1e9:0.2f} seconds')

  return self_play_buffer

def train(net, optim):
  global buffer

  print("  - training -", len(buffer))

  for epoch in range(1):
    print("epoch", epoch)
    # sample 50% of the buffer
    for batch in random.sample(buffer, len(buffer) // 3): # TODO: actually sample and use batches,
      game, probabilities, value = batch
      if DEBUG:
        print("  game", game.to_play)
        game.print_board()
        print("probabilities", probabilities, "value", value)
      value = torch.tensor([[value]], dtype=torch.float32) # add batch dimension

      net.zero_grad() # zeroes the gradient buffers of all parameters. is that necessary?

      encoded_board = board2state(game)
      predicted_policy, predicted_value = net(encoded_board)

      loss = F.cross_entropy(predicted_policy, probabilities) + F.mse_loss(predicted_value, value)
      loss.backward()

      optim.step()

def battle(old_net, new_net):
  NUM_EVALUATION_GAMES = 50
  old_wins = 0
  draws = 0
  new_wins = 0

  for game_num in range(NUM_EVALUATION_GAMES):
    game = TicTacToe()
    old_net_player = 1 if game_num % 2 == 1 else -1
    print("old net player", old_net_player, "game_num", game_num)

    # TODO: reuse the tree for each net.
    while not game.is_terminal():
      use_old_net = game.to_play == old_net_player
      action, _ = run_mcts(game=game, net=old_net if use_old_net else new_net, num_searches=NUM_SEARCHES, temperature=0, explore=False)
      game = game.next_state(action=action)
      # game.print_board()
      # print()

    winner = game.get_winner()
    if winner == old_net_player: old_wins += 1
    elif winner == 0: draws += 1
    else: new_wins += 1

    print('  eval play game', game_num, "winner", winner, "old player won", winner == old_net_player)

  return old_wins, draws, new_wins

def predict(net):
  encoded_board = board2state(TicTacToe())
  t0 = time.monotonic_ns()
  _ = net(encoded_board)
  if (time.monotonic_ns() - t0) / 1e6 > 10: print("prediction took too long", (time.monotonic_ns() - t0) / 1e6, "ms")

  self_play_buffer = self_play(net)
  return self_play_buffer


def main():
  global buffer

  net = Network()
  optim = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)

  # load model and optimizer if they exist
  if os.path.exists('model.pt') and os.path.exists('optimizer.pt'):
    print("loading model and optimizer...")
    net.load_state_dict(torch.load('model.pt'))
    optim.load_state_dict(torch.load('optimizer.pt'))

  """
  if torch.backends.mps.is_available():
    # device = torch.device("mps") # this seems to be much slower than cpu
    device = torch.device("cpu")
  else:
    device = torch.device("cpu")
  net.to(device)
  """

  for iteration in range(NUM_ITERATIONS):
    # copy the network
    net_copy = copy.deepcopy(net)
    optim_copy = copy.deepcopy(optim)

    print("iteration", iteration)
    with multiprocessing.Pool(NUM_SELF_PLAYERS) as pool:
      t0 = time.monotonic_ns()
      results = pool.map(predict, [net]*NUM_SELF_PLAYERS)
      for result in results:
        buffer.extend(result)
      sec = (time.monotonic_ns() - t0) / 1e9
      print("got ", len(buffer), "games in", sec, "seconds") 
    train(net, optim)

    print("\n"*10, "after training")
    _, root = run_mcts(game=TicTacToe(), net=net, num_searches=NUM_SEARCHES, temperature=0)
    root.visualize()

    # exit()

    # play against previous version
    old_wins, draws, new_wins = battle(old_net=net_copy, new_net=net)
    print("old wins", old_wins, "draws", draws, "new wins", new_wins)
    if new_wins / (old_wins + new_wins) >= 0.55: # new model wins more than 55% of the time
      print(" >>> choose NEW model !!!")

      # save model and optimizer
      torch.save(net.state_dict(), 'model.pt')
      torch.save(optim.state_dict(), 'optimizer.pt')

      # keep data from last iteration
      buffer = buffer[-(NUM_GAMES_SELF_PLAY * NUM_SELF_PLAYERS):]
    else:
      print(" >>> choose OLD model !!!")
      net = net_copy
      optim = optim_copy


if __name__ == "__main__":
  main()
