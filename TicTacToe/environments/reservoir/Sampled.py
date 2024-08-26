from Game import Game
from environments.reservoir.ReservoirGameEnvironment import \
    ReservoirGameEnvironment


class SampledReservoirGameEnvironment(ReservoirGameEnvironment):
    name = "Sampled"

    def __init__(self, game: Game = Game(), parity=True):
        super().__init__(game, sample=1, parity=parity)


class SampledNoParityReservoirGameEnvironment(SampledReservoirGameEnvironment):
    name = "Sampled_NP"

    def __init__(self, game: Game = Game()):
        super().__init__(game, parity=False)


class Sampled10ReservoirGameEnvironment(ReservoirGameEnvironment):
    name = "Sampled10"

    def __init__(self, game: Game = Game()):
        super().__init__(game, sample=10, parity=True)


class Sampled10NoParityReservoirGameEnvironment(ReservoirGameEnvironment):
    name = "Sampled10_NP"

    def __init__(self, game: Game = Game()):
        super().__init__(game, sample=10, parity=False)
