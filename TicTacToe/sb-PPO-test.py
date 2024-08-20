from stable_baselines3 import PPO
from SBTicTacToeEnv import SBTicTacToeEnv
from TicTacToe import evaluate
from TicTacToe import SimpleTicTacToe as TicTacToe

# Load the trained model
env = SBTicTacToeEnv(TicTacToe())
sb_model = PPO.load("ppo_tictactoe", env=env)
evaluate(env, lambda obs: sb_model.predict(obs)[0])
