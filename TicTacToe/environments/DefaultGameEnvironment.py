from Game import TicTacToeGame
from environments.GameEnvironment import TicTacToeEnv
import numpy as np


class SimpleTicTacToe(TicTacToeEnv):
    def __init__(self, game: TicTacToeGame = TicTacToeGame()):
        super().__init__(game)

    def encoded_board(self, current_player):
        return self.game.board.flatten().astype(np.float32) * current_player

    @property
    def input_count(self):
        return self.game.board_size * self.game.board_size

    @property
    def min_input_value(self):
        return -1

    @property
    def max_input_value(self):
        return 1
