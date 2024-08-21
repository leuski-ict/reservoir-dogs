from sb3_contrib import MaskablePPO
from SBMaskableTicTacToeEnv import SBMaskableTicTacToeEnv
from TicTacToe import *
from SBTicTacToeEnv import *
from TicTacToe import SimpleTicTacToe as TicTacToe
from sb3_contrib.common.maskable.utils import get_action_masks

# env = SBMaskableTicTacToeEnv(TicTacToe())
# # Load the trained model
# sb_model = MaskablePPO.load("maskable_ppo_tictactoe", env=env)
# evaluate(env, lambda obs: sb_model.predict(
#     obs, action_masks=get_action_masks(env))[0])

tournament(
    # TicTacToeRandomAgent(),
    SBTicTacToeMaskableAgent(MaskablePPO.load("maskable_x_ppo_tictactoe_f"),
                             TicTacToeFloatBits),
    # TicTacToeRandomAgent()
    SBTicTacToeMaskableAgent(MaskablePPO.load("maskable_o_ppo_tictactoe_f"),
                             TicTacToeFloatBits),
)
