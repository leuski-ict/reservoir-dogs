import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AbstractNN(BaseFeaturesExtractor):
    def forward(self, observations):
        return self.nn(observations)


class DefaultNN(AbstractNN):
    def __init__(self, observation_space, features_dim):
        super(DefaultNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.nn = nn.Sequential(
            nn.Linear(n_input_channels, features_dim),
        )


class SimpleNN(AbstractNN):
    def __init__(self, observation_space, features_dim):
        super(SimpleNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.nn = nn.Sequential(
            nn.Linear(n_input_channels, features_dim),
        )


class OneLayerNN(AbstractNN):
    def __init__(self, observation_space, features_dim):
        super(OneLayerNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        middle_1_channels = 32
        self.nn = nn.Sequential(
            nn.Linear(n_input_channels, middle_1_channels),
            nn.Linear(middle_1_channels, features_dim),
        )


class OneLayerWithTanhNN(AbstractNN):
    def __init__(self, observation_space, features_dim):
        super(OneLayerWithTanhNN, self).__init__(
            observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        middle_1_channels = 32
        self.nn = nn.Sequential(
            nn.Linear(n_input_channels, middle_1_channels),
            nn.Tanh(),
            nn.Linear(middle_1_channels, features_dim),
        )


class TwoLayerWithTanhSoftmaxNN(AbstractNN):
    def __init__(self, observation_space, features_dim):
        super(TwoLayerWithTanhSoftmaxNN, self).__init__(
            observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        middle_1_channels = 32
        self.nn = nn.Sequential(
            nn.Linear(n_input_channels, middle_1_channels),
            nn.Tanh(),
            nn.Linear(middle_1_channels, middle_1_channels),
            nn.Tanh(),
            nn.Linear(middle_1_channels, features_dim),
            nn.Softmax()
        )


class TwoLayerNN(AbstractNN):
    def __init__(self, observation_space, features_dim):
        super(TwoLayerNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        middle_1_channels = 32
        self.nn = nn.Sequential(
            nn.Linear(n_input_channels, middle_1_channels),
            nn.Linear(middle_1_channels, middle_1_channels),
            nn.Linear(middle_1_channels, features_dim),
        )
