from sb3_contrib.common.envs import InvalidActionEnvDiscrete

from environments.StableBaselineEnvironment import StableBaselineEnvironment
from agents.Agent import AbstractAgent
from environments.GameEnvironment import GameEnvironment


class MaskableStableBaselineEnvironment(StableBaselineEnvironment,
                                        InvalidActionEnvDiscrete):
    def __init__(self, tic_tac_toe: GameEnvironment, player,
                 opponent: AbstractAgent = None):
        StableBaselineEnvironment.__init__(self, tic_tac_toe, player, opponent)
        InvalidActionEnvDiscrete.__init__(self)
        self.action_space = self.make_action_space()
        self.observation_space = self.make_observation_space()
