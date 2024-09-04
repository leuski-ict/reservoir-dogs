from Game import Game
from environments.reservoir.ReservoirGameEnvironment import \
    ReservoirGameEnvironment


class MeanReservoirGameEnvironment(ReservoirGameEnvironment):
    name = "Mean"

    def __init__(self, game: Game = Game(), parity=2):
        super().__init__(game, sample=0, parity=parity)


class MeanNoParityReservoirGameEnvironment(MeanReservoirGameEnvironment):
    name = "Mean_NP"

    def __init__(self, game: Game = Game()):
        super().__init__(game, parity=0)


class MeanNoParity1ReservoirGameEnvironment(MeanReservoirGameEnvironment):
    name = "Mean_NP1"

    def __init__(self, game: Game = Game()):
        super().__init__(game, parity=1)
