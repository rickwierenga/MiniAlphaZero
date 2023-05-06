from abc import ABC, abstractmethod
import sys
from typing import Optional

if sys.version_info >= (3, 11):
  from typing import Self
else:
  from typing_extensions import Self


class Game(ABC):
  @abstractmethod
  def print_board(self) -> None:
    """ Print the board in a human-readable format """

  @abstractmethod
  def is_terminal(self) -> bool:
    """ Return True if the game is over """

  @abstractmethod
  def next_state(self, action, player) -> Self:
    """ Return a new Game object with the given action played """

  def get_winner(self) -> Optional[int]:
    """ Return the winner of the game """
  
  @abstractmethod
  def get_legal_moves(self) -> list:
    """ Return a list of legal moves """

  @abstractmethod
  def hash(self) -> int:
    pass
