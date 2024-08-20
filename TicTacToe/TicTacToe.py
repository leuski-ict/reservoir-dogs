import numpy as np
from random import randrange

# Assume this table converts 4-bit sequences to float values
CONDUCTANCE_TABLE = {
    (0, 0, 0, 0): 0.0,
    (0, 0, 0, 1): 0.1,
    (0, 0, 1, 0): 0.2,
    (0, 0, 1, 1): 0.3,
    (0, 1, 0, 0): 0.4,
    (0, 1, 0, 1): 0.5,
    (0, 1, 1, 0): 0.6,
    (0, 1, 1, 1): 0.7,
    (1, 0, 0, 0): 0.8,
    (1, 0, 0, 1): 0.9,
    (1, 0, 1, 0): 1.0,
    (1, 0, 1, 1): 1.1,
    (1, 1, 0, 0): 1.2,
    (1, 1, 0, 1): 1.3,
    (1, 1, 1, 0): 1.4,
    (1, 1, 1, 1): 1.5,
}


def bit_to_float(bits):
    return CONDUCTANCE_TABLE[tuple(bits)]


# Tic-Tac-Toe game environment
class TicTacToe:
    def __init__(self, board_size=3):
        self.board_size = board_size
        self.winner = None
        self.done = None
        self.board = None
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        # x, y = divmod(randrange(self.board_size * self.board_size),
        #               self.board_size)
        # self.board[x, y] = 1
        self.done = False
        self.winner = None
        return self.encoded_board(self.current_player), self.done

    @property
    def current_player(self):
        # if count_nonzero == 0, player = 1
        # ... 1, player = -1
        # ...
        return 1 - 2 * (np.count_nonzero(self.board.flatten()) % 2)

    @property
    def input_count(self):
        return 6

    @property
    def min_input_value(self):
        return min(CONDUCTANCE_TABLE.values())

    @property
    def max_input_value(self):
        return max(CONDUCTANCE_TABLE.values())

    @property
    def output_count(self):
        return 9

    # noinspection PyMethodMayBeStatic
    def max_episode_timesteps(self):
        return 1000

    # noinspection PyMethodMayBeStatic
    def states(self):
        # Define the state space (6-dimensional for the bit pattern encoding)
        return dict(type='float', shape=(self.input_count,))

    # noinspection PyMethodMayBeStatic
    def actions(self):
        # Define the action space (9 discrete actions, one for each cell)
        return dict(type='int', num_values=self.output_count)

    def step(self, player, action):
        x, y = action
        # Check if the action is valid (the cell is empty
        # and the game is not finished)
        if self.done:
            reward = self.get_reward(player)
        elif self.board[x, y] == 0:
            # Place the player's mark on the board
            self.board[x, y] = player
            self._check_winner()
            reward = self.get_reward(player)
        else:
            self.done = True
            reward = -10

        return self.encoded_board(-player), self.done, reward

    def execute(self, actions):
        # Translate action into board coordinates
        action = divmod(actions, self.board_size)
        return self.step(self.current_player, action)

    def _check_winner(self):
        # Iterate over both players (1 for 'X', -1 for 'O')
        for player in [1, -1]:
            # Check rows and columns for a win
            if (any((self.board == player).all(axis=0))
                    or any((self.board == player).all(axis=1))
                    or np.all(np.diag(self.board) == player)
                    or np.all(np.diag(np.fliplr(self.board)) == player)):
                self.done = True
                self.winner = player
                return
        # Check for a draw (no empty cells)
        if not any(0 in row for row in self.board):
            self.done = True
            self.winner = 0

    def get_reward(self, current_player):
        if self.winner == current_player:
            return 1  # Reward for winning
        elif self.winner == 0:
            return 0.5  # Reward for draw
        elif self.done:
            return -1  # Penalize for losing
        else:
            return 0  # No reward if game is not done

    def available_actions(self):
        # Return a list of available actions (empty cells)
        return [(x, y)
                for x in range(self.board_size)
                for y in range(self.board_size)
                if self.board[x, y] == 0]

    def encoded_board(self, current_player):
        encoded_board = []
        for player in [current_player, -current_player]:
            for i in range(3):
                position_bits = []
                count = 0
                for j in range(3):
                    if self.board[i, j] == player:
                        position_bits.append(1)
                        count += 1
                    else:
                        position_bits.append(0)
                position_bits.append(count % 2)
                encoded_board.append(bit_to_float(position_bits))
        return np.array(encoded_board, dtype=np.float32)


class SimpleTicTacToe(TicTacToe):
    def __init__(self, board_size=3):
        super().__init__(board_size)

    def encoded_board(self, current_player):
        return self.board.flatten().astype(np.float32) * current_player

    @property
    def input_count(self):
        return self.board_size * self.board_size

    @property
    def min_input_value(self):
        return -1

    @property
    def max_input_value(self):
        return 1
