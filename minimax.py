import numpy  as np
from game import Game
from util import battle

# Store global state in a cache
cache = {}

def minimax(game: Game, depth=0, max_depth=None):
  """ Perform minimax search to find the optimal move. """
  # Check if we've already seen this state
  if game.hash() in cache:
    best_moves, value = cache[game.hash()]
    # Randomly choose between moves of equal value
    best_move = np.random.choice(best_moves)
    return best_move, value

  if game.is_terminal():
    value = {
      0: 0,
      game.to_play: 1,
      -game.to_play: -1
    }[game.get_winner()]
    cache[game.hash()] = [None], value
    return None, value

  if max_depth is not None and depth >= max_depth:
    return None, 0

  # Find the maximum score by recursively searching each possible move
  best_value = -np.inf
  best_moves = []
  for move in game.get_legal_moves():
    next_game = game.next_state(move)
    _, value = minimax(next_game, depth=depth+1, max_depth=max_depth)
    value = -value # flip because we're looking from the other player's perspective
    if value >= best_value:
      best_value = value
      best_moves.append(move)
  if len(best_moves) == 0:
    best_moves.append(None) # TODO needed?

  cache[game.hash()] = best_moves, best_value
  # Randomly choose between moves of equal value
  best_move = np.random.choice(best_moves)
  return best_move, best_value


if __name__ == "__main__":
  #from tictactoe import TicTacToe
  #game = TicTacToe()
  from connect4 import Connect4
  game = Connect4()
  battle(game, minimax, minimax)
