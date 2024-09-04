from Game import Game
from environments.reservoir.ReservoirGameEnvironment import \
    ReservoirGameEnvironment


class SampledReservoirGameEnvironment(ReservoirGameEnvironment):
    name = "Sampled"

    def __init__(self, game: Game = Game(), parity=2):
        super().__init__(game, sample=1, parity=parity)


class SampledNoParityReservoirGameEnvironment(SampledReservoirGameEnvironment):
    name = "Sampled_NP"

    def __init__(self, game: Game = Game()):
        super().__init__(game, parity=0)


class SampledNoParity1ReservoirGameEnvironment(SampledReservoirGameEnvironment):
    name = "Sampled_NP1"

    def __init__(self, game: Game = Game()):
        super().__init__(game, parity=1)


class Sampled10ReservoirGameEnvironment(ReservoirGameEnvironment):
    name = "Sampled10"

    def __init__(self, game: Game = Game()):
        super().__init__(game, sample=10, parity=2)


class Sampled10NoParityReservoirGameEnvironment(ReservoirGameEnvironment):
    name = "Sampled10_NP"

    def __init__(self, game: Game = Game()):
        super().__init__(game, sample=10, parity=0)
