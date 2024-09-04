import numpy as np
from scipy.stats import norm

from Game import Game
from environments.reservoir.ReservoirGameEnvironment import \
    ReservoirGameEnvironment, CONDUCTANCE_TABLE


class Decoding(ReservoirGameEnvironment):
    name = "Decoding"

    def __init__(self, game: Game = Game()):
        super().__init__(game, sample=1, parity=2)

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
        board_values = [0 for _ in range(self.input_count)]
        for player in [current_player, -current_player]:
            for i in range(self.game.board.size):
                bits = ReservoirGameEnvironment.bits_with_parity_for_row_on(
                    self.game.board, player, i, self.parity, piece=self.piece)
                value = ReservoirGameEnvironment.encode_4_bits(
                    bits, self.sample)

                for key in CONDUCTANCE_TABLE:
                    if key not in Decoding.distributions:
                        mean_and_error = CONDUCTANCE_TABLE[key]
                        distribution = norm(
                            loc=mean_and_error[0], scale=mean_and_error[1] / 2)
                        scale = distribution.pdf(mean_and_error[0])
                        Decoding.distributions[
                            key] = (distribution, scale)
                    dist_n_scale = Decoding.distributions[key]
                    val = dist_n_scale[0].pdf(value) / (12 * dist_n_scale[1])
                    for index, bit in enumerate(key):
                        if bit == 1:
                            board_values[i * len(key) + index] += val * player
        return np.array(board_values, dtype=np.float32)
