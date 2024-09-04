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
    def __init__(self, game: Game, sample: int, parity: int, piece: int = 0):
        super().__init__(game)
        self.sample = sample
        self.parity = parity
        self.piece = piece

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
    def bits_for_row_on(board: Board, player, row: int, piece=0):
        position_bits = []
        count = 0
        for j in range(board.size):
            if board[row, j] == player:
                position_bits.append(piece)
                count += 1
            else:
                position_bits.append(1 - piece)
        return position_bits, count

    @staticmethod
    def bits_with_parity_for_row_on(
            board: Board, player, row: int, parity, piece):
        position_bits, count = ReservoirGameEnvironment.bits_for_row_on(
            board, player, row, piece=piece)
        if parity == 1 or parity == 0:
            position_bits.append(parity)
        else:
            position_bits.append(count % 2)
        return position_bits

    @staticmethod
    def encode_all_bits(bits, sample: int, padding=0):
        bits_tail = len(bits) % 4
        if bits_tail != 0:
            bits = bits + [padding for _ in range(4 - bits_tail)]
        encoded_bits = []
        # len(bits) is divisible by 4.
        for index in range(0, len(bits), 4):
            encoded = ReservoirGameEnvironment.encode_4_bits(
                bits[index:index + 4], sample)
            encoded_bits += encoded
            # print(bits[index:index + 4], encoded)
        # print("")
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
                    self.game.board, player, row_index, self.parity,
                    piece=self.piece)
        return ReservoirGameEnvironment.encode_all_bits(bits, self.sample)
