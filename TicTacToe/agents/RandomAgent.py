import random

from Game import TicTacToeGame
from agents.Agent import TicTacToeAgent


class TicTacToeRandomAgent(TicTacToeAgent):
    def __init__(self):
        super().__init__()

    def make_move(self, game: TicTacToeGame, player=None) -> None:
        if game.done:
            return
        action = random.choice(game.available_actions())
        game.make_move(action, player)
