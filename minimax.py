import numpy  as np
from game import Game
from util import battle


def minimax(game: Game, to_play: int, cache = {}):
  """ Perform minimax search to find the optimal move. """

  # Check if we've already seen this state
  board_hash = hash(str(game.board) + str(to_play))
  if board_hash in cache:
    return cache[board_hash]

  if game.is_terminal():
    return None, {
      0: 0,
      to_play: 1,
      -to_play: -1
    }[game.get_winner()]

  # Find the maximum score by recursively searching each possible move
  max_score = -np.inf
  best_move = None
  for move in game.get_legal_moves():
    next_state = game.next_state(move, to_play)
    _, score = minimax(next_state, -to_play)
    score = -score # flip because we're looking from the other player's perspective
    if score > max_score:
      best_move = move
      max_score = score

  cache[board_hash] = best_move, max_score

  return best_move, max_score


if __name__ == "__main__":
  from tictactoe import TicTacToe
  game = TicTacToe()
  battle(game, minimax, minimax)
