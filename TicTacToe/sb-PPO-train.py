from stable_baselines3 import PPO
from environments.StableBaselineEnvironment import StableBaselineEnvironment, \
    check_env
from environments.DefaultGameEnvironment import DefaultGameEnvironment

model = PPO('MlpPolicy',
            check_env(StableBaselineEnvironment(DefaultGameEnvironment(), 1)),
            verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_tictactoe")
