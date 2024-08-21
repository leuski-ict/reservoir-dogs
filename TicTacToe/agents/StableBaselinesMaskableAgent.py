from agents.StableBaselinesAgent import SBTicTacToeAgent


class SBTicTacToeMaskableAgent(SBTicTacToeAgent):
    def select_action(self, player):
        observations = self.tic_tac_toe.encoded_board(player)
        action_masks = self.tic_tac_toe.action_masks()
        return self.model.predict(observations, deterministic=True,
                                  action_masks=action_masks)[0]
