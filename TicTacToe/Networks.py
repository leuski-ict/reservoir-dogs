import torch as th
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
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


class TwoLayerReluNN(AbstractNN):
    def __init__(self, observation_space, features_dim):
        super(TwoLayerReluNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        middle_1_channels = 64
        middle_2_channels = 128
        self.nn = nn.Sequential(
            nn.Linear(n_input_channels, middle_1_channels),
            nn.ReLU(),
            nn.Linear(middle_1_channels, middle_2_channels),
            nn.ReLU(),
            nn.Linear(middle_2_channels, features_dim),
        )


class TwoLayerRReluNN(AbstractNN):
    def __init__(self, observation_space, features_dim):
        super(TwoLayerRReluNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        middle_1_channels = 64
        middle_2_channels = 64
        self.nn = nn.Sequential(
            nn.Linear(n_input_channels, middle_1_channels),
            nn.RReLU(),
            nn.Linear(middle_1_channels, middle_2_channels),
            nn.RReLU(),
            nn.Linear(middle_2_channels, features_dim),
        )


class TwoLayerRRelu16x16NN(AbstractNN):
    def __init__(self, observation_space, features_dim):
        super(TwoLayerRRelu16x16NN, self).__init__(observation_space,
                                                   features_dim)
        n_input_channels = observation_space.shape[0]
        middle_1_channels = 16
        middle_2_channels = 16
        self.nn = nn.Sequential(
            nn.Linear(n_input_channels, middle_1_channels),
            nn.RReLU(),
            nn.Linear(middle_1_channels, middle_2_channels),
            nn.RReLU(),
            nn.Linear(middle_2_channels, features_dim),
        )


class TwoLayerRRelu32x32NN(AbstractNN):
    def __init__(self, observation_space, features_dim):
        super(TwoLayerRRelu32x32NN, self).__init__(observation_space,
                                                   features_dim)
        n_input_channels = observation_space.shape[0]
        middle_1_channels = 32
        middle_2_channels = 32
        self.nn = nn.Sequential(
            nn.Linear(n_input_channels, middle_1_channels),
            nn.RReLU(),
            nn.Linear(middle_1_channels, middle_2_channels),
            nn.RReLU(),
            nn.Linear(middle_2_channels, features_dim),
        )


class TwoLayerRRelu16x32NN(AbstractNN):
    def __init__(self, observation_space, features_dim):
        super(TwoLayerRRelu16x32NN, self).__init__(observation_space,
                                                   features_dim)
        n_input_channels = observation_space.shape[0]
        middle_1_channels = 16
        middle_2_channels = 32
        self.nn = nn.Sequential(
            nn.Linear(n_input_channels, middle_1_channels),
            nn.RReLU(),
            nn.Linear(middle_1_channels, middle_2_channels),
            nn.RReLU(),
            nn.Linear(middle_2_channels, features_dim),
        )


class BitDecoder1NN(AbstractNN):
    def __init__(self, observation_space, features_dim):
        super(BitDecoder1NN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        middle_1_channels = 64
        middle_2_channels = 9
        self.nn = nn.Sequential(
            nn.Linear(n_input_channels, middle_1_channels),
            nn.RReLU(),
            nn.Linear(middle_1_channels, middle_1_channels),
            nn.RReLU(),
            nn.Linear(middle_1_channels, middle_2_channels),
            nn.Tanh(),
            nn.Linear(middle_2_channels, features_dim),
        )


class CustomLSTMNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomLSTMNetwork, self).__init__(observation_space, features_dim)
        self.lstm = nn.LSTM(
            input_size=observation_space.shape[1],
            hidden_size=features_dim, batch_first=True)
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # batch_size = observations.size(0)
        lstm_out, _ = self.lstm(observations)
        return self.flatten(lstm_out[:, -1, :])


class CustomLSTMPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomLSTMPolicy, self).__init__(
            *args, **kwargs,
            features_extractor_class=CustomLSTMNetwork,
            features_extractor_kwargs=dict(features_dim=128))
