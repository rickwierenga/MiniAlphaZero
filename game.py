from abc import ABC, abstractmethod
import sys
from typing import Optional

if sys.version_info >= (3, 11):
  from typing import Self
else:
  from typing_extensions import Self


import numpy as np


class Game(ABC):
  @abstractmethod
  def get_user_action(self) -> np.array:
    """ Get the user's action """

  @abstractmethod
  def print_board(self) -> None:
    """ Print the board in a human-readable format """

  @abstractmethod
  def is_terminal(self) -> bool:
    """ Return True if the game is over """

  @abstractmethod
  def next_state(self, action, player) -> Self:
    """ Return a new Game object with the given action played """

  def get_available_actions(self) -> list:
    """ Return a list of legal moves """

  def get_winner(self) -> Optional[int]:
    """ Return the winner of the game """
  
  def get_reward(self, player) -> int:
    return {
      player: 1,
      -player: -1,
      0: 0
    }[player]
