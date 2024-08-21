import random

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


class TicTacToeGame:
    def __init__(self, board_size=3):
        self.board_size = board_size
        self.winner = None
        self.done = None
        self.board = None
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.done = False
        self.winner = None

    @property
    def current_player(self):
        # if count_nonzero == 0, player = 1
        # ... 1, player = -1
        # ...
        return 1 - 2 * (np.count_nonzero(self.board.flatten()) % 2)

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

    def available_actions(self):
        # Return a list of available actions (empty cells)
        return [(x, y)
                for x in range(self.board_size)
                for y in range(self.board_size)
                if self.board[x, y] == 0]

    @staticmethod
    def board_row_as_string(board, row_index):
        return "".join(["X" if column == 1 else "O" if column == -1 else "."
                        for column in board[row_index]])

    def row_as_string(self, row_index):
        return TicTacToeGame.board_row_as_string(self.board, row_index)

    def board_as_string(self):
        return "\n".join([self.row_as_string(row_index)
                          for row_index in range(self.board_size)])

    def make_move(self, action, player=None) -> bool:
        if player is None:
            player = self.current_player
        if type(action) is not tuple:
            action = divmod(action, self.board_size)
        x, y = action
        if self.done:
            return False
        elif self.board[x, y] == 0:
            # Place the player's mark on the board
            self.board[x, y] = player
            self._check_winner()
            return True
        else:
            return False


class TicTacToeEnv:
    def __init__(self, game: TicTacToeGame):
        self.game = game

    def reset(self):
        self.game.reset()
        return self.encoded_board(self.game.current_player), self.game.done

    def encoded_board(self, current_player):
        raise NotImplementedError

    @property
    def input_count(self):
        raise NotImplementedError

    @property
    def min_input_value(self):
        raise NotImplementedError

    @property
    def max_input_value(self):
        raise NotImplementedError

    @property
    def output_count(self):
        return self.game.board_size * self.game.board_size

    def step(self, action, player=None):
        if player is None:
            player = self.game.current_player
        # Check if the action is valid (the cell is empty
        # and the game is not finished)
        if self.game.done:
            reward = self.get_reward(player)
        elif self.game.make_move(action, player):
            reward = self.get_reward(player)
        else:
            reward = -10
        return self.encoded_board(-player), self.game.done, reward

    def get_reward(self, current_player):
        if self.game.winner == current_player:
            return 1  # Reward for winning
        elif self.game.winner == 0:
            return 0.5  # Reward for draw
        elif self.game.done:
            return -1  # Penalize for losing
        else:
            return 0  # No reward if game is not done

    def action_masks(self):
        result = [x == 0 for x in self.game.board.flatten()]
        return result


# Tic-Tac-Toe game environment
class TicTacToeFloatBits(TicTacToeEnv):
    def __init__(self, game: TicTacToeGame):
        super().__init__(game)

    @property
    def input_count(self):
        return 2 * self.game.board_size

    @property
    def min_input_value(self):
        return min(CONDUCTANCE_TABLE.values())

    @property
    def max_input_value(self):
        return max(CONDUCTANCE_TABLE.values())

    def encoded_board(self, current_player):
        encoded_board = []
        for player in [current_player, -current_player]:
            for i in range(self.game.board_size):
                position_bits = []
                count = 0
                for j in range(self.game.board_size):
                    if self.game.board[i, j] == player:
                        position_bits.append(1)
                        count += 1
                    else:
                        position_bits.append(0)
                position_bits.append(count % 2)
                encoded_board.append(bit_to_float(position_bits))
        return np.array(encoded_board, dtype=np.float32)


class SimpleTicTacToe(TicTacToeEnv):
    def __init__(self, game: TicTacToeGame):
        super().__init__(game)

    def encoded_board(self, current_player):
        return self.game.board.flatten().astype(np.float32) * current_player

    @property
    def input_count(self):
        return self.game.board_size * self.game.board_size

    @property
    def min_input_value(self):
        return -1

    @property
    def max_input_value(self):
        return 1


def board_history_as_string(boards: [[[int]]]) -> str:
    return "\n".join([" ".join([TicTacToeGame.board_row_as_string(board, row)
                                for board in boards])
                      for row in range(len(boards[0][0]))])


def evaluate(env, action_fn, num_games=100):
    print("starting evaluation")
    wins = {1: 0, -1: 0, 0: 0, None: 0}
    for _ in range(num_games):
        obs = env.reset()[0]
        history = []
        while not env.tic_tac_toe.done:
            action = action_fn(obs)
            obs = env.step(action)[0]
            history.append(env.tic_tac_toe.board.copy())
            if env.tic_tac_toe.done:
                print("winner", env.tic_tac_toe.winner)
                print(board_history_as_string(history))
                wins[env.tic_tac_toe.winner] += 1
    print("Wins:", wins)


def tournament(x_agent, o_agent, num_games=100):
    print("starting tournament")
    wins = {1: 0, -1: 0, 0: 0, None: 0}
    game = TicTacToeGame()
    for _ in range(num_games):
        game.reset()
        action = random.choice(game.available_actions())
        game.make_move(action, 1)

        history = []
        while not game.done:
            (x_agent if game.current_player == 1 else o_agent).make_move(game)
            history.append(game.board.copy())
            if game.done:
                print("winner", game.winner)
                print(board_history_as_string(history))
                wins[game.winner] += 1
    print("Wins:", wins)


class TicTacToeAgent:
    def __init__(self):
        pass

    def make_move(self, game: TicTacToeGame, player=None) -> None:
        raise NotImplementedError


class TicTacToeRandomAgent(TicTacToeAgent):
    def __init__(self):
        super().__init__()

    def make_move(self, game: TicTacToeGame, player=None) -> None:
        if game.done:
            return
        action = random.choice(game.available_actions())
        game.make_move(action, player)


