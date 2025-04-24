from Game import Game
from Board import Board
from environments.GameEnvironment import GameEnvironment
from environments.reservoir.ConductanceTable import \
    ConductanceTable, original_table


class ReservoirGameEnvironment(GameEnvironment):
    def __init__(self, game: Game, sample: int,
                 parity: int, piece: int = 0,
                 table: ConductanceTable = original_table):
        super().__init__(game)
        self.sample = sample
        self.parity = parity
        self.piece = piece
        self.table = table

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

    @property
    def input_count(self):
        return 2 * self.game.board.size * max(1, self.sample)

    @property
    def min_input_value(self):
        return self.table.table_min

    @property
    def max_input_value(self):
        return self.table.table_max

    def encoded_board(self, current_player):
        bits = []
        for player in [current_player, -current_player]:
            for row_index in range(self.game.board.size):
                bits += ReservoirGameEnvironment.bits_with_parity_for_row_on(
                    self.game.board, player, row_index, self.parity,
                    piece=self.piece)
        return self.table.encode_all_bits(bits, self.sample)
