from Game import Game


class AbstractAgent:
    def __init__(self):
        pass

    def make_move(self, game: Game, player=None) -> None:
        raise NotImplementedError
