from Game import TicTacToeGame
from agents.Agent import TicTacToeAgent


class SBTicTacToeAgent(TicTacToeAgent):
    def __init__(self, model, env_type):
        super().__init__()
        self.model = model
        self.tic_tac_toe = None
        self.env_type = env_type

    def make_move(self, game: TicTacToeGame, player=None) -> None:
        if game.done:
            return
        if self.tic_tac_toe is None or self.tic_tac_toe.game is not game:
            self.tic_tac_toe = self.env_type(game)
        if player is None:
            player = game.current_player
        action = self.select_action(player)
        game.make_move(action, player)

    def select_action(self, player):
        observations = self.tic_tac_toe.encoded_board(player)
        return self.model.predict(observations, deterministic=True)[0]

