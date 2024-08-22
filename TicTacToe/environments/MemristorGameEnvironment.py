from Game import TicTacToeGame
from environments.GameEnvironment import TicTacToeEnv
import numpy as np

# Assume this table converts 4-bit sequences to float values
CONDUCTANCE_TABLE = {
    (0, 0, 0, 0): (0.6344719, 0.00795708),
    (0, 0, 0, 1): (1.295158482, 0.018508253),
    (0, 0, 1, 0): (0.732387046, 0.00857042),
    (0, 0, 1, 1): (1.513310788, 0.019179896),
    (0, 1, 0, 0): (0.653960243, 0.008624319),
    (0, 1, 0, 1): (1.403799596, 0.024181049),
    (0, 1, 1, 0): (0.812054838, 0.007432739),
    (0, 1, 1, 1): (1.690582836, 0.031750193),
    (1, 0, 0, 0): (0.633037114, 0.013263677),
    (1, 0, 0, 1): (1.350882082, 0.023577157),
    (1, 0, 1, 0): (0.834046672, 0.024497146),
    (1, 0, 1, 1): (1.631329556, 0.04620028),
    (1, 1, 0, 0): (0.708502949, 0.01601286),
    (1, 1, 0, 1): (1.496165371, 0.021980554),
    (1, 1, 1, 0): (0.949045017, 0.015482592),
    (1, 1, 1, 1): (1.796519291, 0.013471359),
}


class TicTacToeFloatBits(TicTacToeEnv):
    def __init__(self, game: TicTacToeGame = TicTacToeGame(),
                 sample: bool = False):
        super().__init__(game)
        self.sample = sample

    @property
    def input_count(self):
        return 2 * self.game.board.size

    @property
    def min_input_value(self):
        return min(CONDUCTANCE_TABLE.values()) * 0.93

    @property
    def max_input_value(self):
        return max(CONDUCTANCE_TABLE.values()) * 1.05

    def encoded_board(self, current_player):
        encoded_board = []
        for player in [current_player, -current_player]:
            for i in range(self.game.board.size):
                position_bits = []
                count = 0
                for j in range(self.game.board.size):
                    if self.game.board[i, j] == player:
                        position_bits.append(1)
                        count += 1
                    else:
                        position_bits.append(0)
                position_bits.append(count % 2)
                value_and_error = CONDUCTANCE_TABLE[tuple(position_bits)]
                if self.sample:
                    value = np.random.normal(
                        value_and_error[0], value_and_error[1])
                else:
                    value = value_and_error[0]
                encoded_board.append(value)
        return np.array(encoded_board, dtype=np.float32)


class TicTacToeFloat(TicTacToeFloatBits):
    def __init__(self, game: TicTacToeGame = TicTacToeGame()):
        super().__init__(game, True)
