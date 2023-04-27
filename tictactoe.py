import numpy as np

from game import Game


class TicTacToe(Game):
  def __init__(self, board=None):
    if board is None:
      self.board = np.zeros((3, 3))
    else:
      self.board = board

  def get_winner(self):
    for i in range(3):
      if np.all(self.board[i,:] == 1) or np.all(self.board[:,i] == 1):
        return 1
      if np.all(self.board[i,:] == -1) or np.all(self.board[:,i] == -1):
        return -1
    if np.all(np.diag(self.board) == 1) or np.all(np.diag(np.fliplr(self.board)) == 1):
      return 1
    if np.all(np.diag(self.board) == -1) or np.all(np.diag(np.fliplr(self.board)) == -1):
      return -1
    if np.count_nonzero(self.board) == 9:
      return 0
    return None

  def print_board(self):
    for i in range(3):
      print("-------------")
      row_str = "| "
      for j in range(3):
        if self.board[i][j] == 1:
          row_str += "X" + " | "
        elif self.board[i][j] == -1:
          row_str += "O" + " | "
        else:
          row_str += " " + " | "
      print(row_str)
    print("-------------")

  def get_legal_moves(self):
    """ List of available moves for current player """
    return list(zip(*np.where(self.board == 0)))

  def next_state(self, action, player):
    """ Create a new board with the given action """
    if self.board[action[0]][action[1]] != 0:
      raise ValueError("Invalid action")
    board = np.copy(self.board)
    row = action[0]
    col = action[1]
    board[row][col] = player
    return TicTacToe(board=board)

  def get_num_moves(self):
    return np.count_nonzero(self.board == 0)

  def is_terminal(self):
    return self.get_winner() is not None

  def get_user_action(self):
    """ Prompt the user for a column to play in """
    while True:
      try:
        row = int(input("Enter a row: "))
        col = int(input("Enter a column: "))
        if not (row, col) in self.get_legal_moves():
          raise ValueError("Invalid move")
        return (row, col)
      except ValueError:
        print("Invalid move")



  def available_moves_mask(self):
    """ Mask of available moves for current player """
    return np.where(self.board == 0, 1, 0)
