from environments.reservoir.ReservoirGameEnvironment import *


class SequenceReservoirGameEnvironment(ReservoirGameEnvironment):
    name = "Sequence"

    def __init__(self, game: Game = Game(), parity=2):
        self.history_length = 4
        self.history = []
        super().__init__(game, sample=1, parity=parity)

    def reset(self):
        self.game.reset()
        self.history = []
        key = (self.game.board.board_x, self.game.board.board_o)
        for _ in range(self.history_length):
            self.history.append((key, self.encoded_board(1)))
        super().reset()

    @property
    def observation_space_shape(self):
        return self.history_length, self.input_count

    @property
    def input_count(self):
        return 2 * self.game.board.size * max(
            1, self.sample)

    def observation(self, current_player):
        key = (self.game.board.board_x, self.game.board.board_o)
        if self.history[-1][0] != key:
            if len(self.history) > 0:
                self.history.pop(0)
            self.history.append((key, self.encoded_board(current_player)))
        return np.array([elem[1] for elem in self.history])
