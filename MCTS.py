# basic MCTS implementation

import random

import numpy as np

random.seed(0)
np.random.seed(0)


class Node:
  def __init__(self, player, state, C, action, parent=None):
    # player who just played the action. this is so we can evaluate the reward for the
    # parent node from the perspective of the player who just played the action
    self.player = player

    self.state = state
    self.action = action
    self.C = C
    self.children = []
    self.visits = 0
    self.reward = 0
    self.parent = parent

  def add_child(self, action):
    # TODO: we switch the player here, think AlphaZero does it differently
    # The player who is gonna play next is the opposite of the player who just played
    child_state = self.state.next_state(action=action, player=-self.player)
    child = Node(player=-self.player, state=child_state, C=self.C, action=action, parent=self)
    self.children.append(child)

  def update(self, reward):
    self.reward += reward
    self.visits += 1

  def ucb(self):
    if self.visits == 0:
      return np.inf
    return self.reward / self.visits + self.C * np.sqrt(np.log(self.parent.visits) / self.visits)

  def is_terminal(self):
    return self.state.is_terminal()

  def get_legal_moves(self):
    return self.state.get_legal_moves()

  def visualize(self, level=0):
    print("    " * level + f"action: {self.action}   visits: {self.visits} reward: {self.reward}  player={self.player}")
    for child in self.children:
      child.visualize(level + 1)

  def __repr__(self):
    return f"Node(action={self.action}   visits={self.visits}, reward={self.reward})"
  
  def win_probability(self):
    if self.visits == 0:
      return -1
    return self.reward / self.visits


class MCTS:
  def __init__(self, state, player, C):
    self.root = Node(state=state, C=C, action=None, player=player)
    self.C = C

  def start(self):
    # we could start by expanding the root node, but the algorithm will do that anyway.
    # the benefit is not computing the value of the root node
    self.expand(self.root)

    for _ in range(10000):
      current = self.select(self.root)

      assert len(current.children) == 0, "I think select always returns a leaf node, so the if statement can be shortened"

      if len(current.children) == 0:
        if current.visits == 0:
          self.rollout(current)
        else:
          # TODO: what should we do if there is no more legal move?
          # For now, we will just skip the expansion and move on to roll out
          if not current.is_terminal():
            self.expand(current)
            current = current.children[0]
          self.rollout(current)
      else:
        # select current according to tree policy
        current = max(current.children, key=lambda child: child.ucb())

    # greedily select the best action
    return max(self.root.children, key=lambda child: child.visits)

  def select(self, node):
    # select child node with highest UCB until we reach a leaf node
    # this is the tree policy
    while not len(node.children) == 0:
      node = max(node.children, key=lambda child: child.ucb())
    return node

  def rollout(self, node):
    # get value of terminal state
    state = node.state

    if not state.is_terminal():
      # simulate game play until terminal state
      simulation_player = node.player

      while True:
        # Take random actions until we reach a terminal state
        action = random.choice(state.get_legal_moves())
        state = state.next_state(action=action, player=simulation_player)

        if state.is_terminal():
          break

        simulation_player = -simulation_player

    # get value of terminal state: did the player from the rollout node win?
    # the reward is for player who started the rollout
    final_state_reward = state.get_reward(node.player)

    # backpropagate reward
    while node is not None:
      node.update(final_state_reward)
      final_state_reward = -final_state_reward # flip the reward for the other player
      node = node.parent

  def expand(self, node):
    # for each available action, add a new state to the tree
    for action in node.get_legal_moves():
      node.add_child(action=action)



def play_against_user(game):
  current_player = 1
  computer_player = 1

  while not game.is_terminal():
    print("\n")
    mcts = MCTS(game, player=current_player, C=1.41)
    game.print_board()
    print("available moves: ", game.get_legal_moves())
    best_move = mcts.start()

    if current_player == computer_player: # computer is player 1
      # I think this prints the best move for the computer
      print("computer move: ", best_move.action, "value", round(best_move.win_probability(), 2))
      game = game.next_state(action=best_move.action, player=current_player)
    else:
      action = game.get_user_action()

      print("computer says: best move", best_move.action, "value", round(best_move.win_probability(), 2))

      game = game.next_state(action=action, player=current_player)

    if game.is_terminal():
      print("\n")
      print("game over")
      print("computer reward", game.get_reward(computer_player))
      print("user reward", game.get_reward(-computer_player))
      print("\n")
      print("==== end ====")
      print("\n")
      break

    current_player = -current_player


from tictactoe import TicTacToe
from connect4 import Connect4

game = TicTacToe()
game = Connect4()
play_against_user(game)
