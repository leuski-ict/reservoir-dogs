import random

from Game import Game
from agents.Agent import AbstractAgent


class RandomAgent(AbstractAgent):
    def __init__(self):
        super().__init__()

    def make_move(self, game: Game, player=None) -> None:
        if game.done:
            return
        action = random.choice(game.available_actions())
        game.make_move(action, player)
