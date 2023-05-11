import numpy as np

from game import Game


class Connect4(Game):
  def __init__(self, board=None, history=None, to_play=1):
    self.board = np.zeros((6, 7), dtype=int) if board is None else board
    self.num_rows, self.num_cols = self.board.shape
    self.history = [] if history is None else history
    self.winning_length = 4
    self.to_play = to_play

  def is_valid_move(self, col):
    if col < 0 or col >= self.num_cols:
      return False
    return self.board[self.num_rows-1][col] == 0

  def get_next_open_row(self, col):
    for r in range(self.num_rows):
      if self.board[r][col] == 0:
        return r
    raise ValueError("Column is full")

  def get_winner(self):
    # check rows
    for r in range(self.num_rows):
      for c in range(self.num_cols - self.winning_length + 1):
        if self.check_sequence(self.board[r, c:c+self.winning_length]):
          return self.board[r][c]

    # check columns
    for c in range(self.num_cols):
      for r in range(self.num_rows - self.winning_length + 1):
        if self.check_sequence(self.board[r:r+self.winning_length, c]):
          return self.board[r][c]

    # check diagonals
    for r in range(self.num_rows - self.winning_length + 1):
      for c in range(self.num_cols - self.winning_length + 1):
        if self.check_sequence(self.board[r:r+self.winning_length, c:c+self.winning_length].diagonal()):
          return self.board[r][c]
        if self.check_sequence(np.fliplr(self.board[r:r+self.winning_length, c:c+self.winning_length]).diagonal()):
          return self.board[r][c+self.winning_length-1]

    return 0

  def check_sequence(self, seq):
    return len(set(seq)) == 1 and seq[0] != 0

  def is_board_full(self):
    return not any(0 in row for row in self.board)

  def print_board(self):
    """ Print a human-readable representation of the board """

    for r in reversed(range(self.num_rows)):
      print("-" * (self.num_cols * 4 + 1))
      row_str = "| "
      for c in range(self.num_cols):
        if self.board[r][c] == 1:
          row_str += "X" + " | "
        elif self.board[r][c] == -1:
          row_str += "O" + " | "
        else:
          row_str += " " + " | "
      print(row_str)
    print("-" * (self.num_cols * 4 + 1))

  def get_legal_moves(self):
    """ Return a list of legal moves """
    if self.is_terminal():
      return []
    return [c for c in range(self.num_cols) if self.is_valid_move(c)]

  def is_terminal(self):
    """ Return True if the game is over """
    return self.get_winner() != 0 or self.is_board_full()

  def next_state(self, action):
    """ Return a new Connect4 object with the given action played """

    if not self.is_valid_move(action):
      raise ValueError("Invalid move")

    new_board = np.copy(self.board)
    row = self.get_next_open_row(action)
    new_board[row][action] = self.to_play
    new_state = Connect4(board=new_board, to_play=-self.to_play, history=self.history + [action])
    return new_state

  def hash(self) -> int:
    return hash(str(self.board) + str(self.to_play))

def get_human_move(game):
  while True:
    try:
      num = int(input("Enter column: "))
      return num, None
    except ValueError:
      print("invalid num")
