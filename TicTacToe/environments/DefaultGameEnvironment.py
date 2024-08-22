from Game import Game
from environments.GameEnvironment import GameEnvironment
import numpy as np


class DefaultGameEnvironment(GameEnvironment):
    name = "Default"

    def __init__(self, game: Game = Game()):
        super().__init__(game)

    def encoded_board(self, current_player):
        return self.game.board.to_array() * current_player

    @property
    def input_count(self):
        return self.game.board.area

    @property
    def min_input_value(self):
        return -1

    @property
    def max_input_value(self):
        return 1
