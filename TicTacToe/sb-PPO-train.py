from stable_baselines3 import PPO
from environments.SBTicTacToeEnv import SBTicTacToeEnv, check_env
from environments.DefaultGameEnvironment import SimpleTicTacToe

model = PPO('MlpPolicy', check_env(SBTicTacToeEnv(SimpleTicTacToe(), 1)),
            verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_tictactoe")
