from TicTacToe import test
from Networks import *
from environments.DefaultGameEnvironment import *
from environments.ReservoirGameEnvironment import *

test(-1, DefaultNN, DefaultGameEnvironment, num_games=10000,
     print_history=False)

for nn in [SimpleNN, OneLayerNN, OneLayerWithTanhNN, TwoLayerNN,
           TwoLayerWithTanhSoftmaxNN]:
    for env in [MeanReservoirGameEnvironment, SampledReservoirGameEnvironment,
                SampledNoParityReservoirGameEnvironment]:
        test(-1, nn, env, num_games=10000,
             print_history=False)
