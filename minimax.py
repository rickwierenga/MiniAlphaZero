import numpy  as np
from game import Game
from util import battle


def minimax(game: Game, cache = None, depth=0, max_depth=None):
  """ Perform minimax search to find the optimal move. """
  # TODO randomize actions when multiple actions have the same value

  if cache is None: cache = {}

  # Check if we've already seen this state
  if game.hash() in cache:
    best_move, value = cache[game.hash()]
    return best_move, value

  if max_depth is not None and depth >= max_depth:
    return None, 0

  if game.is_terminal():
    value = {
      0: 0,
      game.to_play: 1,
      -game.to_play: -1
    }[game.get_winner()]
    cache[game.hash()] = None, value
    return None, value

  # Find the maximum score by recursively searching each possible move
  best_value = -np.inf
  best_move = None
  for move in game.get_legal_moves():
    next_game = game.next_state(move)
    _, value = minimax(next_game, cache, depth=depth+1, max_depth=max_depth)
    value = -value # flip because we're looking from the other player's perspective
    if value > best_value:
      best_move = move
      best_value = value

  cache[game.hash()] = best_move, best_value

  return best_move, best_value


if __name__ == "__main__":
  #from tictactoe import TicTacToe
  #game = TicTacToe()
  from connect4 import Connect4
  game = Connect4()
  battle(game, minimax, minimax)
