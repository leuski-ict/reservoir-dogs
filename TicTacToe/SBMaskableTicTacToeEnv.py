from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from SBTicTacToeEnv import SBTicTacToeEnv
from TicTacToe import *


class SBMaskableTicTacToeEnv(SBTicTacToeEnv, InvalidActionEnvDiscrete):
    def __init__(self, tic_tac_toe: TicTacToeEnv, player,
                 opponent: TicTacToeAgent = None):
        SBTicTacToeEnv.__init__(self, tic_tac_toe, player, opponent)
        InvalidActionEnvDiscrete.__init__(self)
        self.action_space = self.make_action_space()
        self.observation_space = self.make_observation_space()
