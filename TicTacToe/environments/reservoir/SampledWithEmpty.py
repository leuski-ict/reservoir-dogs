from Game import Game
from environments.reservoir.ReservoirGameEnvironment import \
    ReservoirGameEnvironment
from environments.reservoir.Sampled import SampledReservoirGameEnvironment


class SampledZeroReservoirGameEnvironment(SampledReservoirGameEnvironment):
    name = "SampledWZ"

    def __init__(self, game: Game = Game(), parity: int = 2):
        super().__init__(game, parity=parity)

    @property
    def input_count(self):
        return 3 * self.game.board.size * max(1, self.sample)

    def encoded_board(self, current_player):
        bits = []
        for player in [current_player, -current_player, 0]:
            for row_index in range(self.game.board.size):
                bits += ReservoirGameEnvironment.bits_with_parity_for_row_on(
                    self.game.board, player, row_index, self.parity,
                    piece=self.piece)
        return self.table.encode_all_bits(bits, self.sample)


class SampledZeroNoParityReservoirGameEnvironment(
        SampledZeroReservoirGameEnvironment):
    name = "SampledWZ_NP"

    def __init__(self, game: Game = Game()):
        super().__init__(game, parity=0)
