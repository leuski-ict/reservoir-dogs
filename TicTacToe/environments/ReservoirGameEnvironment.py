from Game import Game
from Board import Board
from environments.GameEnvironment import GameEnvironment
import numpy as np
from scipy.stats import norm

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


class MeanReservoirGameEnvironment(GameEnvironment):
    name = "Mean"

    def __init__(self, game: Game = Game(),
                 sample: bool = False,
                 parity: bool = True):
        super().__init__(game)
        self.sample = sample
        self.parity = parity

    @property
    def input_count(self):
        return 2 * self.game.board.size

    @property
    def min_input_value(self):
        return CONDUCTANCE_TABLE_MIN

    @property
    def max_input_value(self):
        return CONDUCTANCE_TABLE_MAX

    @staticmethod
    def value_for_row(board: Board, player, row: int, parity, sample) -> float:
        position_bits = []
        count = 0
        for j in range(board.size):
            if board[row, j] == player:
                position_bits.append(1)
                count += 1
            else:
                position_bits.append(0)
        if parity:
            position_bits.append(count % 2)
        else:
            position_bits.append(0)
        value_and_error = CONDUCTANCE_TABLE[tuple(position_bits)]
        if sample:
            value = np.random.normal(
                value_and_error[0], value_and_error[1])
            return max(min(value, CONDUCTANCE_TABLE_MAX),
                       CONDUCTANCE_TABLE_MIN)
        else:
            return value_and_error[0]

    def encoded_board(self, current_player):
        encoded_board = []
        for player in [current_player, -current_player]:
            for i in range(self.game.board.size):
                value = MeanReservoirGameEnvironment.value_for_row(
                    self.game.board, player, i, self.parity, self.sample)
                encoded_board.append(value)
        return np.array(encoded_board, dtype=np.float32)


class SampledReservoirGameEnvironment(MeanReservoirGameEnvironment):
    name = "Sampled"

    def __init__(self, game: Game = Game()):
        super().__init__(game, sample=True, parity=True)


class SampledNoParityReservoirGameEnvironment(MeanReservoirGameEnvironment):
    name = "Sampled_NP"

    def __init__(self, game: Game = Game()):
        super().__init__(game, sample=True, parity=False)


class MeanNoParityReservoirGameEnvironment(MeanReservoirGameEnvironment):
    name = "Mean_NP"

    def __init__(self, game: Game = Game()):
        super().__init__(game, sample=False, parity=False)


class DecodingReservoirGameEnvironment(GameEnvironment):
    name = "Decoding"

    def __init__(self, game: Game = Game()):
        super().__init__(game)
        self.parity = True
        self.sample = True

    @property
    def input_count(self):
        return 12

    @property
    def min_input_value(self):
        return -1

    @property
    def max_input_value(self):
        return 1

    distributions = {}

    def encoded_board(self, current_player):
        encoded_board = [0 for _ in range(self.input_count)]
        for player in [current_player, -current_player]:
            for i in range(self.game.board.size):
                value = MeanReservoirGameEnvironment.value_for_row(
                    self.game.board, player, i, self.parity, self.sample)

                for key in CONDUCTANCE_TABLE:
                    if key not in DecodingReservoirGameEnvironment.distributions:
                        mean_and_error = CONDUCTANCE_TABLE[key]
                        distribution = norm(
                            loc=mean_and_error[0], scale=mean_and_error[1] / 2)
                        scale = distribution.pdf(mean_and_error[0])
                        DecodingReservoirGameEnvironment.distributions[
                            key] = (distribution, scale)
                    distribution_and_scale = \
                        DecodingReservoirGameEnvironment.distributions[key]
                    val = distribution_and_scale[0].pdf(value) / \
                          (12 * distribution_and_scale[1])
                    for index, bit in enumerate(key):
                        if bit == 1:
                            encoded_board[i * len(key) + index] += val * player
        return np.array(encoded_board, dtype=np.float32)
