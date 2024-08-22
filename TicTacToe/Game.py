from Board import Board


class Game:
    def __init__(self, board_size=3):
        self.winner = None
        self.done = None
        self.board = Board(board_size)
        self.reset()

    def reset(self):
        self.board.clear()
        self.done = False
        self.winner = None

    @property
    def current_player(self):
        # if count_nonzero == 0, player = 1
        # ... 1, player = -1
        # ...
        return 1 - 2 * (self.board.count_filled_squares() % 2)

    def _check_winner(self):
        # Check for a draw (no empty cells)
        if self.board.is_full():
            self.done = True
            self.winner = 0
            return
        # Iterate over both players (1 for 'X', -1 for 'O')
        for player in [1, -1]:
            # Check rows and columns for a win
            if self.board.check_winner(player):
                self.done = True
                self.winner = player
                return

    def available_actions(self):
        # Return a list of available actions (empty cells)
        return self.board.available_moves()

    def make_move(self, action, player=None) -> bool:
        if player is None:
            player = self.current_player
        if self.done:
            return False
        elif self.board[action] == 0:
            # Place the player's mark on the board
            self.board[action] = player
            self._check_winner()
            return True
        else:
            return False

    def take_back_move(self, action):
        self.board[action] = 0
        self.done = False
        self.winner = None
