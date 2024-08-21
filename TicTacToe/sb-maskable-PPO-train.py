import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from TicTacToe import *
from SBTicTacToeEnv import *
from SBMaskableTicTacToeEnv import SBMaskableTicTacToeEnv


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


def file_name(player):
    if player == 1:
        return "maskable_x_ppo_tictactoe_f"
    else:
        return "maskable_o_ppo_tictactoe_f"


def opponent(player):
    try:
        model = MaskablePPO.load(file_name(player))
        return SBTicTacToeMaskableAgent(model, TicTacToeFloatBits)
    except ValueError:
        return TicTacToeRandomAgent()


def get_model(player):
    env = check_env(SBMaskableTicTacToeEnv(
        TicTacToeFloatBits(TicTacToeGame()), player))
    try:
        return MaskablePPO.load(file_name(player), env=env), env
    except FileNotFoundError:
        return MaskablePPO('MlpPolicy', env, policy_kwargs=policy_kwargs,
                           verbose=1), env


def train(player):
    model, env = get_model(player)
    env.opponent = opponent(-player)
    model.learn(total_timesteps=10000)
    model.save(file_name(player))


def train_multiple():
    model_x, env_x = get_model(1)
    model_o, env_o = get_model(-1)
    env_x.opponent = SBTicTacToeMaskableAgent(model_o, TicTacToeFloatBits)
    env_o.opponent = SBTicTacToeMaskableAgent(model_x, TicTacToeFloatBits)
    for _ in range(1000):
        model_x.learn(total_timesteps=100)
        model_o.learn(total_timesteps=100)
    model_x.save(file_name(1))
    model_o.save(file_name(-1))


train_multiple()
