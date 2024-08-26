from agents.StableBaselinesAgent import StableBaselineAgent


class StableBaselineMaskableAgent(StableBaselineAgent):
    def select_action(self, player):
        observations = self.tic_tac_toe.observation(player)
        action_masks = self.tic_tac_toe.action_masks()
        return self.model.predict(observations, deterministic=True,
                                  action_masks=action_masks)[0]

    @staticmethod
    def file_name(suffix):
        return "../maskable_ppo_tictactoe_" + suffix
