from sb3_contrib import MaskablePPO

from TicTacToe import *
from agents.MinimaxAgent import TicTacToeMinimaxAgent
from agents.StableBaselinesMaskableAgent import SBTicTacToeMaskableAgent
from environments.MemristorGameEnvironment import TicTacToeFloatBits

# env = SBMaskableTicTacToeEnv(TicTacToe())
# # Load the trained model
# sb_model = MaskablePPO.load("maskable_ppo_tictactoe", env=env)
# evaluate(env, lambda obs: sb_model.predict(
#     obs, action_masks=get_action_masks(env))[0])

# tournament(
#     # TicTacToeRandomAgent(),
#     SBTicTacToeMaskableAgent(MaskablePPO.load("../maskable_x_ppo_tictactoe_f2"),
#                              TicTacToeFloatBits),
#     # TicTacToeRandomAgent()
#     # SBTicTacToeMaskableAgent(MaskablePPO.load("maskable_o_ppo_tictactoe_f2"),
#     #                          TicTacToeFloatBits),
#     # TicTacToeMinimaxAgent(),
#     TicTacToeMinimaxAgent(),
# )

tournament(
    # TicTacToeRandomAgent(),
    SBTicTacToeMaskableAgent(MaskablePPO.load("../maskable_x_ppo_tictactoe_f3"),
                             TicTacToeFloatBits),
    # TicTacToeRandomAgent()
    TicTacToeMinimaxAgent(),
    # SBTicTacToeMaskableAgent(MaskablePPO.load("../maskable_o_ppo_tictactoe_f3"),
    #                          TicTacToeFloatBits),
    # TicTacToeMinimaxAgent(),
)
