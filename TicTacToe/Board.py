import copy

import numpy as np


class Board:
    _winning_combinations_cache = {}

    def __init__(self, size=3):
        assert size * size <= 64, "Board size must be less than or equal to 64"
        self.size = size
        self.board_x = 0  # Bit mask for 'X' (1)
        self.board_o = 0  # Bit mask for 'O' (-1)
        area = size * size
        if area not in Board._winning_combinations_cache:
            Board._winning_combinations_cache[
                area] = Board._generate_winning_combinations(size)
        self.winning_combinations = Board._winning_combinations_cache[area]

    def copy(self):
        return copy.copy(self)

    @property
    def area(self):
        return self.size * self.size

    def clear(self):
        self.board_x = 0
        self.board_o = 0

    def __getitem__(self, index):
        if type(index) is tuple:
            index = index[0] * self.size + index[1]
        if self.board_x & (1 << index):
            return 1
        elif self.board_o & (1 << index):
            return -1
        else:
            return 0

    def __setitem__(self, index, value):
        if type(index) is tuple:
            index = index[0] * self.size + index[1]
        if value == 1:
            self.board_x |= (1 << index)
            self.board_o &= ~(1 << index)
        elif value == -1:
            self.board_o |= (1 << index)
            self.board_x &= ~(1 << index)
        else:
            self.board_x &= ~(1 << index)
            self.board_o &= ~(1 << index)

    def check_winner(self, player):
        board = self.board_x if player == 1 else self.board_o
        for combo in self.winning_combinations:
            if (board & combo) == combo:
                return True
        return False

    def available_moves(self):
        return [i for i in range(self.area) if
                not (self.board_x | self.board_o) & (1 << i)]

    def is_full(self):
        return (self.board_x | self.board_o) == (1 << self.area) - 1

    def count_filled_squares(self):
        # Count the number of set bits in board_x and board_o
        return self.board_x.bit_count() + self.board_o.bit_count()

    def to_array(self):
        return np.array([self[i] for i in range(self.area)], dtype=np.float32)

    @staticmethod
    def _generate_winning_combinations(size):
        # Generate winning combinations for rows, columns, and diagonals
        combinations = []

        # Rows
        for row in range(size):
            combo = 0
            for col in range(size):
                combo |= (1 << (row * size + col))
            combinations.append(combo)

        # Columns
        for col in range(size):
            combo = 0
            for row in range(size):
                combo |= (1 << (row * size + col))
            combinations.append(combo)

        # Diagonals
        combo = 0
        for i in range(size):
            combo |= (1 << (i * size + i))
        combinations.append(combo)

        combo = 0
        for i in range(size):
            combo |= (1 << (i * size + (size - 1 - i)))
        combinations.append(combo)

        return combinations

    @staticmethod
    def _symbol(value):
        if value == 1:
            return 'X'
        elif value == -1:
            return 'O'
        else:
            return '_'

    def row_as_string(self, row_index):
        return "".join([Board._symbol(self[row_index, column])
                        for column in range(self.size)])

    def __str__(self):
        return "\n".join([self.row_as_string(row_index)
                          for row_index in range(self.size)])


class BoardList(list):
    def __str__(self):
        if len(self) == 0:
            return "no boards"
        return "\n".join([" ".join([board.row_as_string(row)
                                    for board in self])
                          for row in range(self[0].size)])
