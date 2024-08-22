from TicTacToe import train_one
from Networks import *
from environments.DefaultGameEnvironment import *
from environments.ReservoirGameEnvironment import *

train_one(-1, DefaultNN, DefaultGameEnvironment)

for nn in [SimpleNN, OneLayerNN, OneLayerWithTanhNN, TwoLayerNN,
           TwoLayerWithTanhSoftmaxNN]:
    for env in [MeanReservoirGameEnvironment, SampledReservoirGameEnvironment,
                SampledNoParityReservoirGameEnvironment]:
        train_one(-1, nn, env)
