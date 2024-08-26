from Game import Game
from agents.Agent import AbstractAgent
from environments.reservoir.ReservoirGameEnvironment import \
    ReservoirGameEnvironment


class StableBaselineAgent(AbstractAgent):
    tic_tac_toe: ReservoirGameEnvironment | None

    def __init__(self, model, env_type):
        super().__init__()
        self.model = model
        self.tic_tac_toe = None
        self.env_type = env_type

    def make_move(self, game: Game, player=None) -> None:
        if game.done:
            return
        if self.tic_tac_toe is None or self.tic_tac_toe.game is not game:
            self.tic_tac_toe = self.env_type(game)
        if player is None:
            player = game.current_player
        action = self.select_action(player)
        game.make_move(action, player)

    def select_action(self, player):
        observations = self.tic_tac_toe.observation(player)
        return self.model.predict(observations, deterministic=True)[0]
