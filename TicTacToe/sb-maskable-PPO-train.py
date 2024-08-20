import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from TicTacToe import SimpleTicTacToe as TicTacToe
from SBTicTacToeEnv import check_env
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

env = check_env(SBMaskableTicTacToeEnv(TicTacToe()))

model = MaskablePPO(
    'MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=100000)
# evaluate_policy(model, env, n_eval_episodes=20, warn=False)
model.save("maskable_ppo_tictactoe")
