from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from SBTicTacToeEnv import SBTicTacToeEnv
from TicTacToe import SimpleTicTacToe as TicTacToe


class SBMaskableTicTacToeEnv(SBTicTacToeEnv, InvalidActionEnvDiscrete):
    def __init__(self, tic_tac_toe: TicTacToe):
        SBTicTacToeEnv.__init__(self, tic_tac_toe)
        InvalidActionEnvDiscrete.__init__(self)
        self.action_space = self.make_action_space()
        self.observation_space = self.make_observation_space()
        self.reset()
