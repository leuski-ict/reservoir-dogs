from agents.StableBaselinesAgent import StableBaselineAgent


class StableBaselineMaskableAgent(StableBaselineAgent):
    def select_action(self, player):
        observations = self.tic_tac_toe.encoded_board(player)
        action_masks = self.tic_tac_toe.action_masks()
        return self.model.predict(observations, deterministic=True,
                                  action_masks=action_masks)[0]

    @staticmethod
    def file_name(player, suffix):
        if player == 1:
            return "../maskable_x_ppo_tictactoe_" + suffix
        else:
            return "../maskable_o_ppo_tictactoe_" + suffix
