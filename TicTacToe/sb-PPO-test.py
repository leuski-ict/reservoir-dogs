from stable_baselines3 import PPO
from environments.StableBaselineEnvironment import StableBaselineEnvironment
from TicTacToe import evaluate
from environments.DefaultGameEnvironment import DefaultGameEnvironment

# Load the trained model
env = StableBaselineEnvironment(DefaultGameEnvironment(), 1)
sb_model = PPO.load("ppo_tictactoe")
evaluate(env, lambda obs: sb_model.predict(obs)[0])
