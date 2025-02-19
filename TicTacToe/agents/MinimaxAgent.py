import random

from Game import Game
from agents.Agent import AbstractAgent


class MinimaxAgent(AbstractAgent):
    _cache = {}

    def __init__(self):
        super().__init__()
        self.epsilon = 0

    def make_move(self, game: Game, player=None) -> None:
        if game.done:
            return
        action = self.best_move(game, player)
        game.make_move(action, player)

    def minimax(self, game: Game, depth, alpha, beta, player,
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
                prediction = self.evaluate_move(move, game, player, depth + 1,
                                                alpha, beta, not is_maximizing)
                max_eval = max(max_eval, prediction)
                alpha = max(alpha, prediction)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in game.available_actions():
                prediction = self.evaluate_move(move, game, player, depth + 1,
                                                alpha, beta, not is_maximizing)
                min_eval = min(min_eval, prediction)
                beta = min(beta, prediction)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_move(self, move, game: Game, player, depth=0,
                      alpha=float('-inf'), beta=float('inf'),
                      is_maximizing=False):
        game.make_move(move)
        prediction = self.minimax(game, depth, alpha, beta, player,
                                  is_maximizing)
        game.take_back_move(move)
        return prediction

    def all_best_move_estimations(self, game: Game, player):
        if player is None:
            player = game.current_player
        key = (game.board.board_x, game.board.board_o, player)
        if key not in MinimaxAgent._cache:
            estimations = [
                (move, self.evaluate_move(move, game, player))
                for move in game.available_actions()]
            max_estimation = max(estimations, key=lambda x: x[1])[1]
            best_moves = [pair for pair in estimations if
                          pair[1] == max_estimation]
            MinimaxAgent._cache[key] = best_moves
        return MinimaxAgent._cache[key]

    def best_move_estimation(self, game: Game, player):
        return random.choice(self.all_best_move_estimations(game, player))

    def best_move(self, game: Game, player):
        if player is None:
            player = game.current_player
        if self.epsilon > 0 and random.random() < self.epsilon:
            return random.choice(game.available_actions())
        est = self.best_move_estimation(game, player)
        return est[0]
