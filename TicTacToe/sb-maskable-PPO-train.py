from TicTacToe import train_one
from Networks import *
from environments.reservoir import *

# train_one(-1, DefaultNN, DefaultGameEnvironment)
# train_one(-1, DefaultNN, DefaultGameEnvironment, suffix="fast-reward-1")

for nn in [
    SimpleNN,
    # OneLayerNN,
    # OneLayerWithTanhNN,
    # TwoLayerNN,
    # TwoLayerWithTanhSoftmaxNN,
    # TwoLayerRReluNN,
    # BitDecoder1NN,
    # TwoLayerRRelu16x16NN,
    # TwoLayerRRelu16x32NN,
    # TwoLayerRRelu32x32NN,
]:
    for env in [
        MeanReservoirGameEnvironment,
        MeanNoParityReservoirGameEnvironment,
        SampledReservoirGameEnvironment,
        SampledNoParityReservoirGameEnvironment,
        # DecodingReservoirGameEnvironment,

        # SampledZeroReservoirGameEnvironment,
        # SampledZeroNoParityReservoirGameEnvironment,
        # SampledRollReservoirGameEnvironment,
        # SampledRoll2ReservoirGameEnvironment,
        # Sampled10ReservoirGameEnvironment,
        # Sampled10NoParityReservoirGameEnvironment,
    ]:
        train_one(-1, nn, env, steps=25_000_000)
