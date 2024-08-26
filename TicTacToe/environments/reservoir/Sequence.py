from environments.reservoir.ReservoirGameEnvironment import *


class SequenceReservoirGameEnvironment(ReservoirGameEnvironment):
    name = "Sequence"

    def __init__(self, game: Game = Game(), parity=True):
        self.history_length = 4
        self.history = []
        super().__init__(game, sample=1, parity=parity)

    def reset(self):
        self.game.reset()
        self.history = []
        super().reset()

    def input_count(self):
        return 2 * self.game.board.size * max(
            1, self.sample) * self.history_length

    def observation(self, current_player):
        return self.encoded_board(current_player)
