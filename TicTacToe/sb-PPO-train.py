from stable_baselines3 import PPO
from TicTacToe import SimpleTicTacToe as TicTacToe
from SBTicTacToeEnv import SBTicTacToeEnv, check_env

model = PPO('MlpPolicy', check_env(SBTicTacToeEnv(TicTacToe())), verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_tictactoe")
