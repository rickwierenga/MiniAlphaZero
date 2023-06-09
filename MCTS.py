import copy
import multiprocessing
import os
import pickle
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from connect4 import Connect4
from game import Game
from resnet import Network

NUM_SEARCHES = 576
NUM_GAMES_SELF_PLAY = 3000 # this will be removed once the self play is always running
NUM_ITERATIONS = 1000 # number of iterations of self play, training, and evaluation
NUM_SELF_PLAYERS = 8
NUM_EPOCHS = 20
NUM_SIMULTANEOUS_SEARCHES = 16
BATCH_SIZE = 128

NUM_SAMPLING_MOVES = 10 # connect 4: 10, tttt: 2
NUM_EVALUATION_GAMES = 100

VIRTUAL_LOSS = 3.1 # use non integer value, to avoid uct from getting 0 visit counts :)

C = 2 # 1.41

DEBUG = os.environ.get("DEBUG", 0)
if DEBUG:
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)

  NUM_SEARCHES = 10
  BATCH_SIZE = 1
  NUM_GAMES_SELF_PLAY = 1
  NUM_SELF_PLAYERS = 1
  NUM_ITERATIONS = 1
  NUM_EPOCHS = 1
  NUM_EVALUATION_GAMES = 1
  NUM_SIMULTANEOUS_SEARCHES = 1


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


def board2state(game):
#   state = torch.zeros((3, 3, 3))
  state = torch.zeros((3, 6, 7))
  board = torch.tensor(game.board)
  state[0, :, :] = board == 1
  state[1, :, :] = board == -1
  state[2, :, :] = game.to_play
  return state


def uct_score(node: Node, parent: Node):
  # C = math.log((parent.visit_count + config.pb_c_base + 1) /
  #                 config.pb_c_base) + config.pb_c_init
  prior_score = C * node.prior * np.sqrt(parent.visit_count) / (1 + node.visit_count) # U
  return -node.get_value() + prior_score # flip value because we're looking from the other player's perspective

def expand(leaf_nodes: List[Node], games: List[Game], net: Network, device):
  # obtain the policy and value from the network
  encoded_boards = torch.stack([board2state(game) for game in games]) #.to(device)
  with torch.no_grad():
    policies, values = net(encoded_boards)

  # for each available action, add a new state to the tree
  for leaf_node, game, policy in zip(leaf_nodes, games, policies):
    for action in game.get_legal_moves():
      leaf_node.add_child(action=action, prior=policy[action]) # action indexes into policy, which is a 3x3 matrix

  return values

def select_action(node: Node, temperature: float):
  if temperature == 0: # strongest move: the one with the highest visit count
    return max(node.children, key=lambda x: node.children[x].visit_count)

  # select an action according to the visit counts
  visit_counts = np.array([child.visit_count for child in node.children.values()])
  visit_counts = visit_counts ** (1 / temperature)
  assert visit_counts.sum() != 0
  visit_counts = visit_counts / visit_counts.sum()
  # hacky, but numpy does not like us selecting from a list of tuples (thinks it's 2d)
  action_index = np.random.choice(len(visit_counts), p=visit_counts)
  action = list(node.children.keys())[action_index]
  return action

def run_mcts(game: Game, net: Network, device = torch.device("cpu"), num_searches: int = None, temperature=None, root=None, explore: bool = True):
  # start by expanding the root node
  if num_searches is None: # Init
    num_searches = NUM_SEARCHES 
  if root is None:
    root = Node(to_play=game.to_play, prior=0)
    expand(leaf_nodes=[root], net=net, games=[game], device=device)

  # add Dirichlet noise to the root node
  epsilon = 0.25
  dir_noise = 0.3
  num_actions = len(root.children)
  noise = torch.distributions.dirichlet.Dirichlet(torch.ones((num_actions,)) * dir_noise).sample()
  for i, node in enumerate(root.children.values()):
    node.prior = (1 - epsilon) * node.prior + epsilon * noise[i]

  for _ in range(num_searches // NUM_SIMULTANEOUS_SEARCHES):
    paths = []
    scratch_games = []

    # Collect 16 paths, a different one using virtual loss. In Real AlphaZero this is parallelized, but that's
    # hard for us because multiprocessing copies memory (overhead + virtual loss does not work), and Python threading
    # is affected by the GIL.
    for _ in range(NUM_SIMULTANEOUS_SEARCHES):
      # select a leaf node according to UCT
      node = root
      path = [node]
      scratch_game = copy.deepcopy(game)
      while node.get_expanded():
        action, node = max(node.children.items(), key=lambda x: uct_score(node=x[1], parent=node))
        path.append(node)
        scratch_game = scratch_game.next_state(action=action) 
        node.visit_count += VIRTUAL_LOSS
        node.value_sum -= VIRTUAL_LOSS
      scratch_games.append(scratch_game)
      paths.append(path)

    # Compute the value and policy for each leaf node in a batch.
    # These are the values from the perspective of who's to play in the leaf node.
    leaf_nodes = [path[-1] for path in paths]
    values = expand(games=scratch_games, leaf_nodes=leaf_nodes, net=net, device=device)

    # backpropagate through each path
    for path, value, scratch_game in zip(paths, values, scratch_games):
      for node in path:
        # if the game is over, the value is the actual reward
        if scratch_game.is_terminal():
          value = scratch_game.get_winner() * scratch_game.to_play
        node.value_sum += value if node.to_play == scratch_game.to_play else -value
        node.visit_count += 1

        if node is not path[0]: # skip the root node
          node.visit_count -= VIRTUAL_LOSS
          node.value_sum += VIRTUAL_LOSS

  # Return the action to play and the action probabilities (normalized visit counts).
  if temperature is None:
    temperature = 1 if len(game.history) < NUM_SAMPLING_MOVES and explore else 0
  action = select_action(root, temperature=temperature)
  if DEBUG:
    root.visualize()
    print("choosing", action)
  return action, root


def self_play(net, game_num, device):
  print("start self play")
  self_play_buffer = []

  t = time.monotonic_ns()
  game = Connect4()
  root = Node(to_play=1, prior=0)
  game_buffer = []

  while not game.is_terminal():
    action, root = run_mcts(root=root, game=game, net=net, num_searches=NUM_SEARCHES, explore=True, device=device)

    # mcts_policy = torch.zeros((3, 3))
    mcts_policy = torch.zeros((7,))
    for a, n in root.children.items():
      mcts_policy[a] = n.visit_count
    mcts_policy = mcts_policy / mcts_policy.sum() # renormalize

    game_buffer.append((game, mcts_policy))

    game = game.next_state(action=action)
    if game_num % 100 == 0:
      game.print_board()
      print(mcts_policy, root.get_value())
      print()

    root = root.children[action] # reuse the tree

  winner = game.get_winner()
  for game, probabilities in game_buffer:
    self_play_buffer.append((game, probabilities, winner * game.to_play)) # trick: multiply by the winner to flip the sign if the winner is -1

  print(f'  self play game {game_num:04} winner {winner:02} took {(time.monotonic_ns() - t) / 1e9:0.2f} seconds')

  return self_play_buffer

def train(net, optim, device, buffer):
  print("  - training -", len(buffer))

  # move the network to the GPU if available
  net = net.to(device)

  num_batches = len(buffer) // BATCH_SIZE

  for epoch in range(NUM_EPOCHS):
    losssum = 0
    t0 = time.monotonic_ns()
    random.shuffle(buffer)
    batches = [buffer[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range(num_batches)]

    for batch in tqdm.tqdm(batches):
      values = torch.tensor([value for _, _, value in batch], dtype=torch.float32).unsqueeze(1)
      probabilities = torch.stack([probabilities for _, probabilities, _ in batch])
      games = [game for game, _, _ in batch]

      if DEBUG:
        print("training batch")
        print("actual values", values)
        print("actual probs", probabilities)
        print("games")
        for game in games:
          game.print_board()
          print("  game to play", game.to_play)
        print()

      # encode board and move probabilities, value
      encoded_boards = torch.stack([board2state(game) for game in games])
      encoded_boards = encoded_boards.to(device)
      probabilities = probabilities.to(device)
      values = values.to(device)

      net.zero_grad() # zeroes the gradient buffers of all parameters. is that necessary?

      predicted_policies, predicted_values = net(encoded_boards)

      loss = F.cross_entropy(predicted_policies, probabilities) + F.mse_loss(predicted_values, values)
      loss.backward()

      optim.step()

      losssum += loss.item()

    print("epoch", epoch, "loss", losssum, "took", (time.monotonic_ns() - t0) / 1e9, "seconds")
  
  # move the network back to the CPU for inference, TODO: check if this is necessary
  net = net.to(torch.device("cpu"))

def _battle_game(old_net, new_net, game_num, device):
  t0 = time.monotonic_ns()
  game = Connect4()
  old_net_player = 1 if game_num % 2 == 1 else -1

  # TODO: reuse the tree for each net.
  while not game.is_terminal():
    use_old_net = game.to_play == old_net_player
    action, _ = run_mcts(game=game, net=old_net if use_old_net else new_net, num_searches=NUM_SEARCHES, temperature=0, explore=True, device=device)
    game = game.next_state(action=action)

  winner = game.get_winner()
  print('  eval play game', game_num, "winner", winner, "old player won", winner == old_net_player, "in", (time.monotonic_ns() - t0) / 1e9, "seconds")
  return {
    old_net_player: -1,
    0: 0,
    -old_net_player: 1
  }[winner]

def battle(old_net, new_net, device):
  old_wins = 0
  draws = 0

  new_wins = 0
  with multiprocessing.Pool(NUM_SELF_PLAYERS) as pool:
    results = pool.starmap(_battle_game, [(old_net, new_net, game_num, device) for game_num in range(NUM_EVALUATION_GAMES)])
    for result in results:
      if result == -1: old_wins += 1
      elif result == 0: draws += 1
      elif result == 1: new_wins += 1

  return old_wins, draws, new_wins


def self_player(net, device):
  results = []
  # net.to(device)
  for game_num in range(NUM_GAMES_SELF_PLAY // NUM_SELF_PLAYERS):
    plays = self_play(net=net, game_num=game_num, device=device)
    results.extend(plays)
  return results


def main():
  print("main")
  SELF_PLAY_GAMES_FN = "self-play-games.pickle"

  buffer = []
  if os.path.exists(SELF_PLAY_GAMES_FN) and not DEBUG:
    print("loading self play games...")
    with open(SELF_PLAY_GAMES_FN, "rb") as f:
      buffer = pickle.load(f)

  MODEL_FN = "connect4-model.pt"
  OPTIMIZER_FN = "connect4-optimizer.pt"

  # net = Network()
  net = Network(board_size=(6, 7), policy_shape=(7,), num_layers=10)
  # check if we can use the MPS backend
  if torch.backends.mps.is_available():
    device = torch.device("mps:0") # this seems to be much slower than cpu
    #device = torch.device("cpu")
  elif torch.cuda.is_available():
    device = torch.device("cuda:0")
  else:
    device = torch.device("cpu")

  # load model and optimizer if they exist
  if os.path.exists(MODEL_FN) and os.path.exists(OPTIMIZER_FN) and not DEBUG:
    print("loading model and optimizer...")
    net.load_state_dict(torch.load(MODEL_FN))

  # move the network to the GPU if available to create the optimizer there, then cpu for multiprocessing
  net.to(device)
  optim = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
  if os.path.exists(MODEL_FN) and os.path.exists(OPTIMIZER_FN) and not DEBUG:
    optim.load_state_dict(torch.load(OPTIMIZER_FN))
  net = net.to(torch.device("cpu")) # back to cpu for multiprocessing

  for iteration in range(6, NUM_ITERATIONS):
    # copy the network
    # new_net = Network()
    new_net = Network(board_size=(6, 7), policy_shape=(7,), num_layers=10)
    new_net.load_state_dict(net.state_dict())
    new_net.to(device)
    new_optim = torch.optim.Adam(new_net.parameters(), lr=0.001, weight_decay=1e-5)
    new_optim.load_state_dict(optim.state_dict())

    print("iteration", iteration)
    with multiprocessing.Pool(NUM_SELF_PLAYERS) as pool:
      t0 = time.monotonic_ns()
      results = pool.starmap(self_player, [(net, device) for _ in range(NUM_SELF_PLAYERS)])
      results = [item for sublist in results for item in sublist] # flatten
      buffer.extend(results)
      sec = (time.monotonic_ns() - t0) / 1e9
      print("got ", len(results), "states in", sec, "seconds") 

    # save self play games
    if not DEBUG:
      with open(SELF_PLAY_GAMES_FN, "wb") as f:
        pickle.dump(buffer, f)

    train(new_net, new_optim, device, buffer)

    if not DEBUG:
      torch.save(new_net.state_dict(), f"connect4-model-{iteration}.pt")
      torch.save(new_optim.state_dict(), f"connect4-optim-{iteration}.pt")

    # play against previous version
    old_wins, draws, new_wins = battle(old_net=net, new_net=new_net, device=device)
    print("old wins", old_wins, "draws", draws, "new wins", new_wins)

    with open("results.txt", "a") as f:
      f.write(f"ITER: {iteration} OW: {old_wins} D: {draws} NW: {new_wins}\n")

    if (old_wins + new_wins) > 0 and new_wins / (old_wins + new_wins) >= 0.55: # new model wins more than 55% of the time
      print(" >>> choose NEW model !!!")

      # save model and optimizer
      torch.save(net.state_dict(), MODEL_FN)
      torch.save(optim.state_dict(), OPTIMIZER_FN)

      # copy the network
      net = new_net
      optim = new_optim
    else:
      print(" >>> choose OLD model !!!")

    # Keep the last 150 000 states only.
    buffer = buffer[-150000:]


if __name__ == "__main__":
  main()
