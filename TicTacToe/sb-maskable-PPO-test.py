from sb3_contrib import MaskablePPO
from SBMaskableTicTacToeEnv import SBMaskableTicTacToeEnv
from TicTacToe import evaluate
from TicTacToe import SimpleTicTacToe as TicTacToe
from sb3_contrib.common.maskable.utils import get_action_masks


env = SBMaskableTicTacToeEnv(TicTacToe())
# Load the trained model
print("loading model", flush=True)
sb_model = MaskablePPO.load("maskable_ppo_tictactoe")
evaluate(env, lambda obs: sb_model.predict(
    obs, action_masks=get_action_masks(env))[0])
