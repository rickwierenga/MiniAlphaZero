import numpy as np

from game import Game


class Connect4(Game):
  def __init__(self):
    self.board = np.zeros((6, 7), dtype=int)
    self.num_rows, self.num_cols = self.board.shape
    self.winning_length = 4

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

  def get_user_action(self):
    """ Prompt the user for a column to play in """
    while True:
      try:
        col = int(input("Enter a column: "))
        if self.is_valid_move(col):
          return col
        print("Invalid move")
      except ValueError:
        print("Invalid move")
    
  def get_legal_moves(self):
    """ Return a list of legal moves """
    return [c for c in range(self.num_cols) if self.is_valid_move(c)]
  
  def is_terminal(self):
    """ Return True if the game is over """
    return self.get_winner() != 0 or self.is_board_full()
  
  def next_state(self, action, player):
    """ Return a new Connect4 object with the given action played """

    if not self.is_valid_move(action):
      raise ValueError("Invalid move")

    new_state = Connect4()
    new_state.board = np.copy(self.board)
    row = self.get_next_open_row(action)
    new_state.board[row][action] = player
    return new_state
 


# def play():
#     game = Connect4()
#     player = 1
#     while True:
#         game.print_board()
#         col = int(input("Player {}'s turn. Enter a column: ".format(player)))
#         if not game.play_move(col, player):
#             print("Invalid move")
#             continue
#         winner = game.get_winner()
#         if winner != 0:
#             game.print_board()
#             print("Player {} wins!".format(winner))
#             break
#         if game.is_board_full():
#             game.print_board()
#             print("Draw!")
#             break
#         player = -player

#     print(game.board)


# if __name__ == "__main__":
#     play()
