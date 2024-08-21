import numpy as np


class TicTacToeGame:
    def __init__(self, board_size=3):
        self.board_size = board_size
        self.winner = None
        self.done = None
        self.board = None
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.done = False
        self.winner = None

    @property
    def current_player(self):
        # if count_nonzero == 0, player = 1
        # ... 1, player = -1
        # ...
        return 1 - 2 * (np.count_nonzero(self.board.flatten()) % 2)

    def _check_winner(self):
        # Iterate over both players (1 for 'X', -1 for 'O')
        for player in [1, -1]:
            # Check rows and columns for a win
            if (any((self.board == player).all(axis=0))
                    or any((self.board == player).all(axis=1))
                    or np.all(np.diag(self.board) == player)
                    or np.all(np.diag(np.fliplr(self.board)) == player)):
                self.done = True
                self.winner = player
                return
        # Check for a draw (no empty cells)
        if not any(0 in row for row in self.board):
            self.done = True
            self.winner = 0

    def available_actions(self):
        # Return a list of available actions (empty cells)
        return [(x, y)
                for x in range(self.board_size)
                for y in range(self.board_size)
                if self.board[x, y] == 0]

    @staticmethod
    def board_row_as_string(board, row_index):
        return "".join(["X" if column == 1 else "O" if column == -1 else "."
                        for column in board[row_index]])

    def row_as_string(self, row_index):
        return TicTacToeGame.board_row_as_string(self.board, row_index)

    def board_as_string(self):
        return "\n".join([self.row_as_string(row_index)
                          for row_index in range(self.board_size)])

    def make_move(self, action, player=None) -> bool:
        if player is None:
            player = self.current_player
        if type(action) is not tuple:
            action = divmod(action, self.board_size)
        x, y = action
        if self.done:
            return False
        elif self.board[x, y] == 0:
            # Place the player's mark on the board
            self.board[x, y] = player
            self._check_winner()
            return True
        else:
            return False

    def take_back_move(self, action):
        if type(action) is not tuple:
            action = divmod(action, self.board_size)
        x, y = action
        self.board[x, y] = 0
        self.done = False
        self.winner = None
