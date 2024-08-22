import random

from Game import TicTacToeGame
from agents.Agent import TicTacToeAgent


class TicTacToeMinimaxAgent(TicTacToeAgent):
    _cache = {}

    def __init__(self):
        super().__init__()
        self.epsilon = 0

    def make_move(self, game: TicTacToeGame, player=None) -> None:
        if game.done:
            return
        action = self.best_move(game, player)
        game.make_move(action, player)

    def minimax(self, game: TicTacToeGame, depth, alpha, beta, player,
                is_maximizing):
        if game.winner == 0:
            return 0
        elif game.winner == player:
            return 1
        elif game.winner is not None:
            return -1

        if is_maximizing:
            max_eval = float('-inf')
            for move in game.available_actions():
                prediction = self.evaluate_move(
                    move, game, depth + 1, alpha, beta, player,
                    not is_maximizing)
                max_eval = max(max_eval, prediction)
                alpha = max(alpha, prediction)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in game.available_actions():
                prediction = self.evaluate_move(
                    move, game, depth + 1, alpha, beta, player,
                    not is_maximizing)
                min_eval = min(min_eval, prediction)
                beta = min(beta, prediction)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_move(self, move, game: TicTacToeGame, depth, alpha, beta,
                      player,
                      is_maximizing):
        game.make_move(move)
        prediction = self.minimax(game, depth + 1, alpha, beta, player,
                                  is_maximizing)
        game.take_back_move(move)
        return prediction

    def best_move(self, game: TicTacToeGame, player):
        if player is None:
            player = game.current_player
        if self.epsilon > 0 and random.random() < self.epsilon:
            return random.choice(game.available_actions())
        key = (game.board.board_x, game.board.board_o, player)
        if key not in TicTacToeMinimaxAgent._cache:
            estimations = [
                (move, self.evaluate_move(move, game, 0, float('-inf'),
                                          float('inf'), player, False))
                for move in game.available_actions()]
            max_estimation = max(estimations, key=lambda x: x[1])[1]
            best_moves = [pair[0] for pair in estimations if
                          pair[1] == max_estimation]
            TicTacToeMinimaxAgent._cache[key] = best_moves
        return random.choice(TicTacToeMinimaxAgent._cache[key])
