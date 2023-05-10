from typing import Any, Callable, Tuple
import time
from game import Game


Action = Any
Agent = Callable[[Game, int], Tuple[Action, float]]


def battle(game: Game, player: Agent, other_player: Agent, print_game=True):
  """ player: 1, other_player: -1 """
  history = []

  while not game.is_terminal():
    if print_game: print("\n")
    if print_game: game.print_board()
    if print_game: print("available moves: ", game.get_legal_moves())

    # loop until a valid move is made
    t0 = time.monotonic_ns()
    while True:
      if game.to_play == 1:
        action, value = player(game)
      else:
        action, value = other_player(game)
      
      if action not in game.get_legal_moves():
        if print_game: print("Invalid move")
        continue
      break

    history.append((game, action, value))

    game = game.next_state(action=action)
    if print_game: print(f"player: {game.to_play} move: {action} value: {value} thinking time: {(time.monotonic_ns() - t0) / 1e6:.3f} ms")

    if game.is_terminal():
      if print_game: print("\n")
      if print_game: game.print_board()
      if print_game: print("GAME OVER")
      winner = game.get_winner()
      if winner == 1:
        if print_game: print(f"player 1 ({player.__name__}) wins")
      elif winner == -1:
        if print_game: print(f"player -1 ({other_player.__name__}) wins")
      else:
        if print_game: print("draw")
      if print_game: print("\n")
      if print_game: print("==== end ====")
      if print_game: print("\n")
      return winner, history
