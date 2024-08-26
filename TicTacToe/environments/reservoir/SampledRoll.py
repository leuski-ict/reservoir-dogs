from Game import Game
from environments.reservoir.ReservoirGameEnvironment import \
    ReservoirGameEnvironment


class SampledRollReservoirGameEnvironment(ReservoirGameEnvironment):
    name = "SampledRoll"

    def __init__(self, game: Game = Game(), parity: bool = True):
        super().__init__(game, sample=1, parity=parity)

    @property
    def input_count(self):
        return self.game.board.area

    def encoded_board(self, current_player):
        bits = []
        # repeat twice
        for _ in range(2):
            # 2 sides
            for player in [current_player, -current_player]:
                # 3 rows
                for row_index in range(self.game.board.size):
                    # 3 columns
                    bits += ReservoirGameEnvironment.bits_for_row_on(
                        self.game.board, player, row_index)[0]
        return ReservoirGameEnvironment.encode_all_bits(bits, self.sample)


class SampledRoll2ReservoirGameEnvironment(ReservoirGameEnvironment):
    name = "SampledRoll2"

    def __init__(self, game: Game = Game(), parity: bool = True):
        super().__init__(game, sample=True, parity=parity)

    @property
    def input_count(self):
        return self.game.board.area * 2

    def encoded_board(self, current_player):
        bits = []
        # 2 sides
        for player in [current_player, -current_player]:
            # 3 rows
            for row_index in range(self.game.board.size):
                # 3 columns
                row_bits = ReservoirGameEnvironment.bits_for_row_on(
                    self.game.board, player, row_index)[0]
                # repeat four times
                for _ in range(4):
                    bits += row_bits
        return ReservoirGameEnvironment.encode_all_bits(bits, self.sample)
