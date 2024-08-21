from Game import TicTacToeGame


class TicTacToeEnv:
    def __init__(self, game: TicTacToeGame = TicTacToeGame()):
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
