from typing import Any, Callable, Tuple
import time
from game import Game


Agent = Callable[[Game, int], Tuple[Any, float]]


def battle(game: Game, player: Agent, other_player: Agent):
  """ player: 1, other_player: -1 """
  to_play = 1

  while not game.is_terminal():
    print("\n")
    game.print_board()
    print("available moves: ", game.get_legal_moves())

    # loop until a valid move is made
    action = None
    t0 = time.monotonic_ns()
    while action not in game.get_legal_moves():
      if to_play == 1:
        action, value = player(game, to_play)
      else:
        action, value = other_player(game, to_play)

    game = game.next_state(action=action, player=to_play)
    print("player", to_play, "move: ", action, "value: ", value, "thinking time: ", (time.monotonic_ns() - t0) / 1e6, "ms")

    if game.is_terminal():
      print("\n")
      game.print_board()
      print("GAME OVER")
      winner = game.get_winner()
      if winner == 1:
        print(f"player 1 ({player.__name__}) wins")
      elif winner == -1:
        print(f"player -1 ({other_player.__name__}) wins")
      else:
        print("draw")
      print("\n")
      print("==== end ====")
      print("\n")
      break

    to_play = -to_play
