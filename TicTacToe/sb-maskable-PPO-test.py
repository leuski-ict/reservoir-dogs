from TicTacToe import *
from Networks import *
from environments.reservoir import *

# test(-1, DefaultNN, DefaultGameEnvironment, suffix="fast-reward-1",
#      num_games=10000,
#      print_history=False)

for nn in [
    SimpleNN,
    OneLayerNN,
    # OneLayerWithTanhNN,
    # TwoLayerNN,
    # TwoLayerWithTanhSoftmaxNN,
    # TwoLayerReluNN,
    # TwoLayerRReluNN,
    # TwoLayerRRelu16x16NN,
    # TwoLayerRRelu16x32NN,
    # TwoLayerRRelu32x32NN,
]:
    for env in [
        # MeanReservoirGameEnvironment,
        # MeanNoParityReservoirGameEnvironment,
        # SampledReservoirGameEnvironment,
        # SampledNoParityReservoirGameEnvironment,
        # SampledZeroReservoirGameEnvironment,
        # SampledZeroNoParityReservoirGameEnvironment,
        # SampledRollReservoirGameEnvironment,
        # SampledRoll2ReservoirGameEnvironment,
        Sampled10ReservoirGameEnvironment,
        Sampled10NoParityReservoirGameEnvironment,
    ]:
        try:
            test(-1, nn, env, num_games=10000,
                 print_history=False)
        except FileNotFoundError:
            pass

# tournament(MinimaxAgent(), MinimaxAgent(), num_games=100, print_history=True)
