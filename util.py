from typing import Any, Callable, Tuple
import time
from game import Game


Action = Any
Agent = Callable[[Game, int], Tuple[Action, float]]


def battle(game: Game, player: Agent, other_player: Agent):
  """ player: 1, other_player: -1 """
  history = []

  while not game.is_terminal():
    print("\n")
    game.print_board()
    print("available moves: ", game.get_legal_moves())

    # loop until a valid move is made
    t0 = time.monotonic_ns()
    while True:
      if game.to_play == 1:
        action, value = player(game)
      else:
        action, value = other_player(game)
      
      if action not in game.get_legal_moves():
        print("Invalid move")
        continue
      break

    history.append((game, action, value))

    print(f"player: {game.to_play} move: {action} value: {value} thinking time: {(time.monotonic_ns() - t0) / 1e6:.3f} ms")
    game = game.next_state(action=action)

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
      return winner, history
