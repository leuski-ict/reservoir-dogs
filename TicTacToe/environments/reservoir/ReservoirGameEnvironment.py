from Game import Game
from Board import Board
from environments.GameEnvironment import GameEnvironment
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
CONDUCTANCE_TABLE_MIN = min(
    [value[0] for value in CONDUCTANCE_TABLE.values()]) * 0.93
CONDUCTANCE_TABLE_MAX = max(
    [value[0] for value in CONDUCTANCE_TABLE.values()]) * 1.05


class ReservoirGameEnvironment(GameEnvironment):
    def __init__(self, game: Game, sample: int, parity: bool):
        super().__init__(game)
        self.sample = sample
        self.parity = parity

    @staticmethod
    def encode_4_bits(bits, sample: int):
        assert len(bits) == 4, "there has to be 4 bits. Got {}".format(
            len(bits))
        value_and_error = CONDUCTANCE_TABLE[tuple(bits)]
        if sample != 0:
            return [
                max(min(
                    np.random.normal(value_and_error[0], value_and_error[1]),
                    CONDUCTANCE_TABLE_MAX), CONDUCTANCE_TABLE_MIN)
                for _ in range(sample)]
        else:
            return [value_and_error[0]]

    @staticmethod
    def bits_for_row_on(board: Board, player, row: int):
        position_bits = []
        count = 0
        for j in range(board.size):
            if board[row, j] == player:
                position_bits.append(1)
                count += 1
            else:
                position_bits.append(0)
        return position_bits, count

    @staticmethod
    def bits_with_parity_for_row_on(board: Board, player, row: int, parity):
        position_bits, count = ReservoirGameEnvironment.bits_for_row_on(
            board, player, row)
        if parity:
            position_bits.append(count % 2)
        else:
            position_bits.append(0)
        return position_bits

    @staticmethod
    def encode_all_bits(bits, sample: int):
        encoded_bits = []
        for index in range(len(bits) // 4):
            encoded_bits += ReservoirGameEnvironment.encode_4_bits(
                bits[4 * index:4 * index + 4], sample)
        if len(bits) % 4 != 0:
            index = 4 * (len(bits) // 4)
            remainder = bits[index:len(bits)] + [
                0 for _ in range(4 - len(bits) + index)]
            encoded_bits += ReservoirGameEnvironment.encode_4_bits(
                remainder, sample)
        return np.array(encoded_bits, dtype=np.float32)

    @property
    def input_count(self):
        return 2 * self.game.board.size * max(1, self.sample)

    @property
    def min_input_value(self):
        return CONDUCTANCE_TABLE_MIN

    @property
    def max_input_value(self):
        return CONDUCTANCE_TABLE_MAX

    def encoded_board(self, current_player):
        bits = []
        for player in [current_player, -current_player]:
            for row_index in range(self.game.board.size):
                bits += ReservoirGameEnvironment.bits_with_parity_for_row_on(
                    self.game.board, player, row_index, self.parity)
        return ReservoirGameEnvironment.encode_all_bits(bits, self.sample)
