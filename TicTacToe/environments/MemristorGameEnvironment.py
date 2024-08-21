from Game import TicTacToeGame
from environments.GameEnvironment import TicTacToeEnv
import numpy as np

# Assume this table converts 4-bit sequences to float values
CONDUCTANCE_TABLE = {
    (0, 0, 0, 0): 0.616063,
    (0, 0, 0, 1): 1.1934,
    (0, 0, 1, 0): 0.80299,
    (0, 0, 1, 1): 1.46703,
    (0, 1, 0, 0): 0.698621,
    (0, 1, 0, 1): 1.319,
    (0, 1, 1, 0): 0.917451,
    (0, 1, 1, 1): 1.63419,
    (1, 0, 0, 0): 0.66731,
    (1, 0, 0, 1): 1.19565,
    (1, 0, 1, 0): 0.928867,
    (1, 0, 1, 1): 1.52327,
    (1, 1, 0, 0): 0.786058,
    (1, 1, 0, 1): 1.39135,
    (1, 1, 1, 0): 1.11136,
    (1, 1, 1, 1): 1.79242,
}


def bit_to_float(bits):
    return CONDUCTANCE_TABLE[tuple(bits)]


class TicTacToeFloatBits(TicTacToeEnv):
    def __init__(self, game: TicTacToeGame = TicTacToeGame()):
        super().__init__(game)

    @property
    def input_count(self):
        return 2 * self.game.board_size

    @property
    def min_input_value(self):
        return min(CONDUCTANCE_TABLE.values()) * 0.93

    @property
    def max_input_value(self):
        return max(CONDUCTANCE_TABLE.values()) * 1.05

    def encoded_board(self, current_player):
        encoded_board = []
        for player in [current_player, -current_player]:
            for i in range(self.game.board_size):
                position_bits = []
                count = 0
                for j in range(self.game.board_size):
                    if self.game.board[i, j] == player:
                        position_bits.append(1)
                        count += 1
                    else:
                        position_bits.append(0)
                position_bits.append(count % 2)
                encoded_board.append(bit_to_float(position_bits))
        return np.array(encoded_board, dtype=np.float32)
