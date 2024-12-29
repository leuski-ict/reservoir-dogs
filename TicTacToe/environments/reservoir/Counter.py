from Game import Game
from environments.reservoir.ReservoirGameEnvironment import \
    ReservoirGameEnvironment


class SampledSpacedCounterGameEnvironment(ReservoirGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    name = "SampledSpacedCounter"

    def __init__(self, game: Game = Game(), sample: int = 1):
        super().__init__(game, sample=sample, parity=2)

    @property
    def input_count(self):
        return self.game.board.area

    def encoded_board(self, current_player):
        bits = []
        # 3 rows
        for row in range(self.game.board.size):
            for col in range(self.game.board.size):
                if self.game.board[row, col] == current_player:
                    bits += [1, 1, 1, 1]
                elif self.game.board[row, col] == -current_player:
                    bits += [0, 0, 0, 0]
                else:
                    bits += [0, 0, 1, 1]
        return self.table.encode_all_bits(bits, self.sample)


class MeanSpacedCounterGameEnvironment(SampledSpacedCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    name = "MeanSpacedCounter"

    def __init__(self, game: Game = Game()):
        super().__init__(game, sample=0)


class SampledDenseCounterGameEnvironment(ReservoirGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    name = "SampledDenseCounter"

    def __init__(self, game: Game = Game(), sample: int = 1):
        super().__init__(game, sample=sample, parity=2)

    @property
    def input_count(self):
        return 7

    def encoded_board(self, current_player):
        base3 = []
        # 3 rows
        for row in range(self.game.board.size):
            for col in range(self.game.board.size):
                if self.game.board[row, col] == current_player:
                    base3 += [2]
                elif self.game.board[row, col] == -current_player:
                    base3 += [0]
                else:
                    base3 += [1]
        base5 = base3_to_base5(base3)
        bits = []
        for element in base5:
            for index in range(element):
                bits += [1]
            for index in range(4-element):
                bits += [0]
        while len(bits) < 28:
            bits += [0]
        return self.table.encode_all_bits(bits, self.sample)


class MeanDenseCounterGameEnvironment(SampledDenseCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    name = "MeanDenseCounter"

    def __init__(self, game: Game = Game()):
        super().__init__(game, sample=0)


def base3_to_base5(base3_array):
    # Step 1: Convert base 3 array to decimal
    decimal = 0
    for digit in base3_array[::-1]:
        decimal = decimal * 3 + digit

    # Step 2: Convert decimal to base 5 array
    if decimal == 0:
        return [0]

    base5_array = []
    while decimal > 0:
        base5_array.append(decimal % 5)
        decimal //= 5

    return base5_array

