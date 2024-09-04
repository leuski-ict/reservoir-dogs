from Game import Game
from environments.reservoir.ReservoirGameEnvironment import \
    ReservoirGameEnvironment


class SampledRollReservoirGameEnvironment(ReservoirGameEnvironment):
    """
    pack each board side as tight as you can, using 1 for presence of a
    piece and 0 for the absence of a piece. Repeat this twice.
    This will make it 3 bit per column x 3 rows x 2 board sides = 18
    And, then 18 x 2 = 36 bits => 36/4 = 9 input numbers.
    This is the same as SampledRoll1ReservoirGameEnvironment, but repeated
    twice.
    """
    name = "SampledRoll"

    def __init__(self, game: Game = Game(), parity: int = 2):
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
                        self.game.board, player, row_index, piece=self.piece)[0]
        return ReservoirGameEnvironment.encode_all_bits(bits, self.sample)


class SampledRoll2ReservoirGameEnvironment(ReservoirGameEnvironment):
    """
    write each row 4 times.
    This will make it 3 bit per column x 4 times x 3 rows x 2 board sides = 72
    => 72/4 = 18 input numbers
    """
    name = "SampledRoll2"

    def __init__(self, game: Game = Game(), parity: int = 2):
        super().__init__(game, sample=1, parity=parity)

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
                    self.game.board, player, row_index, piece=self.piece)[0]
                # repeat four times
                for _ in range(4):
                    bits += row_bits
        return ReservoirGameEnvironment.encode_all_bits(bits, self.sample)


class SampledRoll3ReservoirGameEnvironment(ReservoirGameEnvironment):
    """
    This is similar to SampledRollReservoirGameEnvironment, but we repeat
    each row bits twice, instead of repeating the whole boards.
    """
    name = "SampledRoll3"

    def __init__(self, game: Game = Game(), parity: int = 2):
        super().__init__(game, sample=1, parity=parity)

    @property
    def input_count(self):
        return self.game.board.area

    def encoded_board(self, current_player):
        bits = []
        # 2 sides
        for player in [current_player, -current_player]:
            # 3 rows
            for row_index in range(self.game.board.size):
                # 3 columns
                row_bits = ReservoirGameEnvironment.bits_for_row_on(
                    self.game.board, player, row_index, piece=self.piece)[0]
                for _ in range(2):
                    bits += row_bits
        return ReservoirGameEnvironment.encode_all_bits(bits, self.sample)


class SampledRowReservoirGameEnvironment(ReservoirGameEnvironment):
    """
    Here we repeat each square bit 4 times -- one square is now uses a single
    4-bit sequence and one input value -- 18 input values. Should perform
    similar to SampledRoll2ReservoirGameEnvironment.
    """
    name = "SampledRow2"

    def __init__(self, game: Game = Game()):
        super().__init__(game, sample=1, parity=0)

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
                    self.game.board, player, row_index, piece=self.piece)[0]
                # repeat four times
                for row_bit in row_bits:
                    for _ in range(4):
                        bits.append(row_bit)
        return ReservoirGameEnvironment.encode_all_bits(bits, self.sample)


class RepeatReservoirGameEnvironment(ReservoirGameEnvironment):
    """
    pack each board side as tight as you can, using 1 for presence of a
    piece and 0 for the absence of a piece.
    This will make it 3 bit per column x 3 rows x 2 board sides = 18
    And, then 18/4 = 5 input numbers
    """
    name = "Repeat"

    def __init__(self, game: Game = Game(), sample: int = 1,
                 piece: int = 1, repeat: int = 1, padding: int = 1):
        super().__init__(game, sample=sample, parity=2, piece=piece)
        self.repeat = repeat
        self.padding = padding

    @property
    def input_count(self):
        return (2 * self.game.board.area * self.repeat - 1) // 4 + 1

    def encoded_board(self, current_player):
        bits = []
        # 2 sides
        for player in [current_player, -current_player]:
            # 3 rows
            for row_index in range(self.game.board.size):
                # 3 columns
                row_bits = ReservoirGameEnvironment.bits_for_row_on(
                    self.game.board, player, row_index, piece=self.piece)[0]
                for _ in range(self.repeat):
                    bits += row_bits
        return ReservoirGameEnvironment.encode_all_bits(
            bits, self.sample, padding=self.padding)


class MeanRoll111ReservoirGameEnvironment(RepeatReservoirGameEnvironment):
    """
    pack each board side as tight as you can, using 1 for presence of a
    piece and 0 for the absence of a piece.
    This will make it 3 bit per column x 3 rows x 2 board sides = 18
    And, then 18/4 = 5 input numbers
    """
    name = "MeanRoll111"

    def __init__(self, game: Game = Game()):
        super().__init__(game, sample=0, repeat=1, piece=1)


class MeanRoll110ReservoirGameEnvironment(RepeatReservoirGameEnvironment):
    """
    pack each board side as tight as you can, using 1 for presence of a
    piece and 0 for the absence of a piece.
    This will make it 3 bit per column x 3 rows x 2 board sides = 18
    And, then 18/4 = 5 input numbers
    """
    name = "MeanRoll110"

    def __init__(self, game: Game = Game()):
        super().__init__(game, sample=0, repeat=1, piece=0)


class SampledRoll111ReservoirGameEnvironment(RepeatReservoirGameEnvironment):
    """
    pack each board side as tight as you can, using 1 for presence of a
    piece and 0 for the absence of a piece.
    This will make it 3 bit per column x 3 rows x 2 board sides = 18
    And, then 18/4 = 5 input numbers
    """
    name = "SampledRoll111"

    def __init__(self, game: Game = Game()):
        super().__init__(game, repeat=1, piece=1)


class SampledRoll121ReservoirGameEnvironment(RepeatReservoirGameEnvironment):
    """
    Same as SampledRoll1ReservoirGameEnvironment, but invert the bits
    """
    name = "SampledRoll121"

    def __init__(self, game: Game = Game()):
        super().__init__(game, repeat=2, piece=1)


class SampledRoll131ReservoirGameEnvironment(RepeatReservoirGameEnvironment):
    """
    Same as SampledRoll1ReservoirGameEnvironment, but invert the bits
    """
    name = "SampledRoll131"

    def __init__(self, game: Game = Game()):
        super().__init__(game, repeat=3, piece=1)


class SampledRoll141ReservoirGameEnvironment(RepeatReservoirGameEnvironment):
    """
    Same as SampledRoll1ReservoirGameEnvironment, but invert the bits
    """
    name = "SampledRoll141"

    def __init__(self, game: Game = Game()):
        super().__init__(game, repeat=4, piece=1)


class SampledRoll110ReservoirGameEnvironment(RepeatReservoirGameEnvironment):
    """
    Same as SampledRoll111ReservoirGameEnvironment, but invert the bits
    """
    name = "SampledRoll110"

    def __init__(self, game: Game = Game()):
        super().__init__(game, repeat=1, piece=0)


class SampledRoll120ReservoirGameEnvironment(RepeatReservoirGameEnvironment):
    """
    Same as SampledRoll121ReservoirGameEnvironment, but invert the bits
    """
    name = "SampledRoll120"

    def __init__(self, game: Game = Game()):
        super().__init__(game, repeat=2, piece=0)


class SampledRoll130ReservoirGameEnvironment(RepeatReservoirGameEnvironment):
    """
    Same as SampledRoll131ReservoirGameEnvironment, but invert the bits
    """
    name = "SampledRoll130"

    def __init__(self, game: Game = Game()):
        super().__init__(game, repeat=3, piece=0)


class SampledRoll140ReservoirGameEnvironment(RepeatReservoirGameEnvironment):
    """
    Same as SampledRoll141ReservoirGameEnvironment, but invert the bits
    """
    name = "SampledRoll140"

    def __init__(self, game: Game = Game()):
        super().__init__(game, repeat=4, piece=0)


class SampledRoll4ReservoirGameEnvironment(ReservoirGameEnvironment):
    """
    this is the same as SampledNoParityReservoirGameEnvironment. But, we use
    the last three bits of each row for the board information, instead of the
    first three bits. The unused (first bit) is set to 0.
    """
    name = "SampledRoll4"

    def __init__(self, game: Game = Game()):
        super().__init__(game, sample=1, parity=0)

    @property
    def input_count(self):
        return 2 * self.game.board.size

    def encoded_board(self, current_player):
        bits = []
        # 2 sides
        for player in [current_player, -current_player]:
            # 3 rows
            for row_index in range(self.game.board.size):
                bits.append(0)
                # 3 columns
                bits += ReservoirGameEnvironment.bits_for_row_on(
                    self.game.board, player, row_index, piece=self.piece)[0]
        return ReservoirGameEnvironment.encode_all_bits(bits, self.sample)


class SampledRoll41ReservoirGameEnvironment(ReservoirGameEnvironment):
    """
    this is the same as SampledRoll4ReservoirGameEnvironment. But, we set the
    unused bit (first bit) to 1.
    """
    name = "SampledRoll41"

    def __init__(self, game: Game = Game()):
        super().__init__(game, sample=1, parity=0)

    @property
    def input_count(self):
        return 2 * self.game.board.size

    def encoded_board(self, current_player):
        bits = []
        # 2 sides
        for player in [current_player, -current_player]:
            # 3 rows
            for row_index in range(self.game.board.size):
                bits.append(1)
                # 3 columns
                bits += ReservoirGameEnvironment.bits_for_row_on(
                    self.game.board, player, row_index, piece=self.piece)[0]
        return ReservoirGameEnvironment.encode_all_bits(bits, self.sample)


class SampledRoll50ReservoirGameEnvironment(ReservoirGameEnvironment):
    """
    this is the same as SampledRollReservoirGameEnvironment. But we have
    an argument to define the extra space between the boards. Here the space
    is 0 == SampledRollReservoirGameEnvironment
    """
    name = "SampledRoll50"

    def __init__(self, game: Game = Game(), space=0):
        super().__init__(game, sample=1, parity=2)
        self.space = space

    @property
    def input_count(self):
        return self.game.board.area + 1

    def encoded_board(self, current_player):
        bits = []
        # repeat twice
        repeat_count = 2
        for repeat_index in range(repeat_count):
            # 2 sides
            for player in [current_player, -current_player]:
                # 3 rows
                for row_index in range(self.game.board.size):
                    # 3 columns
                    bits += ReservoirGameEnvironment.bits_for_row_on(
                        self.game.board, player, row_index, piece=self.piece)[0]
            if repeat_index < (repeat_count - 1):
                for _ in range(self.space):
                    bits.append(0)
        for _ in range(4 - self.space):
            bits.append(0)
        return ReservoirGameEnvironment.encode_all_bits(bits, self.sample)


class SampledRoll51ReservoirGameEnvironment(
    SampledRoll50ReservoirGameEnvironment):
    """
    Add one extra bit between the boards.
    """
    name = "SampledRoll51"

    def __init__(self, game: Game = Game()):
        super().__init__(game, space=1)


class SampledRoll52ReservoirGameEnvironment(
    SampledRoll50ReservoirGameEnvironment):
    """
    Add two extra bits between the boards. Because each board takes 18 bits,
    this equivalent to SampledRoll1ReservoirGameEnvironment, repeated twice.
    """
    name = "SampledRoll52"

    def __init__(self, game: Game = Game()):
        super().__init__(game, space=2)


class SampledRoll53ReservoirGameEnvironment(
    SampledRoll50ReservoirGameEnvironment):
    """
    Add three extra bits between the boards.
    """
    name = "SampledRoll53"

    def __init__(self, game: Game = Game()):
        super().__init__(game, space=3)
