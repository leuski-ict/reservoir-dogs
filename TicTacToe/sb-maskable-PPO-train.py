import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO

from agents.StableBaselinesMaskableAgent import SBTicTacToeMaskableAgent
from environments.SBTicTacToeEnv import *
from environments.SBMaskableTicTacToeEnv import SBMaskableTicTacToeEnv
from agents.MinimaxAgent import TicTacToeMinimaxAgent
from agents.RandomAgent import TicTacToeRandomAgent
from environments.MemristorGameEnvironment import TicTacToeFloatBits
import os
from stable_baselines3.common.monitor import Monitor

log_dir = "../tmp/"
os.makedirs(log_dir, exist_ok=True)


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Linear(n_input_channels, features_dim)
        )

    def forward(self, observations):
        return self.cnn(observations)


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=9)
)


def opponent(player, suffix):
    try:
        model = MaskablePPO.load(
            SBTicTacToeMaskableAgent.file_name(player, suffix))
        return SBTicTacToeMaskableAgent(model, TicTacToeFloatBits)
    except ValueError:
        return TicTacToeRandomAgent()


def get_model(player, suffix):
    our_env = SBMaskableTicTacToeEnv(TicTacToeFloatBits(), player)
    env = check_env(Monitor(our_env, log_dir))
    #    callback = EvalCallback(eval_freq=1000, log_path=log_dir)
    try:
        return MaskablePPO.load(
            SBTicTacToeMaskableAgent.file_name(player, suffix),
            env=env), our_env
    except FileNotFoundError:
        return MaskablePPO('MlpPolicy', env, policy_kwargs=policy_kwargs,
                           verbose=1), env


def train(player, suffix):
    model, env = get_model(player, suffix)
    env.opponent = TicTacToeMinimaxAgent()
    model.learn(total_timesteps=1000000)
    model.save(SBTicTacToeMaskableAgent.file_name(player, suffix))


def train_multiple(suffix):
    model_x, env_x = get_model(1, suffix)
    model_o, env_o = get_model(-1, suffix)
    env_x.opponent = SBTicTacToeMaskableAgent(model_o, TicTacToeFloatBits)
    env_o.opponent = SBTicTacToeMaskableAgent(model_x, TicTacToeFloatBits)
    for iteration in range(10000):
        model_x.learn(total_timesteps=100)
        model_o.learn(total_timesteps=100)
        if iteration % 100 == 0:
            model_x.save(SBTicTacToeMaskableAgent.file_name(1, suffix))
            model_o.save(SBTicTacToeMaskableAgent.file_name(-1, suffix))
    model_x.save(SBTicTacToeMaskableAgent.file_name(1, suffix))
    model_o.save(SBTicTacToeMaskableAgent.file_name(-1, suffix))


# train_multiple()
train(-1, "simple")
