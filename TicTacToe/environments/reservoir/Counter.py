from Game import Game
from environments.reservoir.ReservoirGameEnvironment import \
    ReservoirGameEnvironment
from environments.reservoir.ConductanceTable import *


class SampledSpacedCounterGameEnvironment(ReservoirGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    name = "SampledSpacedCounter"

    def __init__(self, game: Game = Game(),
                 sample: int = 1, table: ConductanceTable = original_table):
        super().__init__(game, sample=sample, parity=2, table=table)

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

    def __init__(self, game: Game = Game(),
                 table: ConductanceTable = original_table):
        super().__init__(game, sample=0, table=table)


class SampledDenseCounterGameEnvironment(ReservoirGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    name = "SampledDenseCounter"

    def __init__(self, game: Game = Game(), sample: int = 1,
                 table: ConductanceTable = original_table):
        super().__init__(game, sample=sample, parity=2, table=table)

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
            for index in range(4 - element):
                bits += [0]
        while len(bits) < 28:
            bits += [0]
        return self.table.encode_all_bits(bits, self.sample)


class MeanDenseCounterGameEnvironment(SampledDenseCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    name = "MeanDenseCounter"

    def __init__(self, game: Game = Game(),
                 table: ConductanceTable = original_table):
        super().__init__(game, sample=0, table=table)


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


class SampledSpacedCounterGameEnvironment1(SampledSpacedCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_1_0_2s
    name = "SampledSpacedCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)


class MeanSpacedCounterGameEnvironment1(MeanSpacedCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_1_0_2s
    name = "MeanSpacedCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)


class SampledSpacedCounterGameEnvironment2(SampledSpacedCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_2_0_5s
    name = "SampledSpacedCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)


class MeanSpacedCounterGameEnvironment2(MeanSpacedCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_2_0_5s
    name = "MeanSpacedCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)


class SampledSpacedCounterGameEnvironment3(SampledSpacedCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_3_1s
    name = "SampledSpacedCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)


class MeanSpacedCounterGameEnvironment3(MeanSpacedCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_3_1s
    name = "MeanSpacedCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)


class SampledSpacedCounterGameEnvironment4(SampledSpacedCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_4_2s
    name = "SampledSpacedCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)


class MeanSpacedCounterGameEnvironment4(MeanSpacedCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_4_2s
    name = "MeanSpacedCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)


class SampledSpacedCounterGameEnvironment5(SampledSpacedCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_5_1s_0_2s
    name = "SampledSpacedCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)


class MeanSpacedCounterGameEnvironment5(MeanSpacedCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_5_1s_0_2s
    name = "MeanSpacedCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)


class SampledSpacedCounterGameEnvironment6(SampledSpacedCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_6_1s_0_5s
    name = "SampledSpacedCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)


class MeanSpacedCounterGameEnvironment6(MeanSpacedCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_6_1s_0_5s
    name = "MeanSpacedCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)


class SampledSpacedCounterGameEnvironment7(SampledSpacedCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_7_2s_0_5s
    name = "SampledSpacedCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)


class MeanSpacedCounterGameEnvironment7(MeanSpacedCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_7_2s_0_5s
    name = "MeanSpacedCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)


class SampledDenseCounterGameEnvironment1(SampledDenseCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_1_0_2s
    name = "SampledDenseCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)


class MeanDenseCounterGameEnvironment1(MeanDenseCounterGameEnvironment):
    """
    encode a value using the number of bits in the reservoir set.
    """
    table = table_5_1_0_2s
    name = "MeanDenseCounter" + "_" + table.name

    def __init__(self, game: Game = Game()):
        super().__init__(game, table=self.__class__.table)
