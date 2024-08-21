from stable_baselines3 import PPO
from environments.SBTicTacToeEnv import SBTicTacToeEnv
from TicTacToe import evaluate
from environments.DefaultGameEnvironment import SimpleTicTacToe

# Load the trained model
env = SBTicTacToeEnv(SimpleTicTacToe(), 1)
sb_model = PPO.load("ppo_tictactoe")
evaluate(env, lambda obs: sb_model.predict(obs)[0])
