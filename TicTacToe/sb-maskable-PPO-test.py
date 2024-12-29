from TicTacToe import *
from Networks import *
from environments.reservoir import *

# test(-1, DefaultNN, DefaultGameEnvironment, suffix="fast-reward-1",
#      num_games=10000,
#      print_history=False)

for nn in [
    SimpleNN,
    # OneLayerNN,
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
        # MeanNoParity1ReservoirGameEnvironment,
        # SampledReservoirGameEnvironment,
        # SampledNoParityReservoirGameEnvironment,
        # SampledNoParity1ReservoirGameEnvironment,
        # DecodingReservoirGameEnvironment,

        # MeanSpacedCounterGameEnvironment,
        # SampledSpacedCounterGameEnvironment,

        MeanDenseCounterGameEnvironment,
        SampledDenseCounterGameEnvironment,

        # MeanRoll110ReservoirGameEnvironment,
        # MeanRoll111ReservoirGameEnvironment,
        # SampledRoll110ReservoirGameEnvironment,
        # SampledRoll111ReservoirGameEnvironment,
        # SampledRoll120ReservoirGameEnvironment,
        # SampledRoll121ReservoirGameEnvironment,
        # SampledRoll130ReservoirGameEnvironment,
        # SampledRoll131ReservoirGameEnvironment,
        # SampledRoll140ReservoirGameEnvironment,
        # SampledRoll141ReservoirGameEnvironment,

        # SampledZeroReservoirGameEnvironment,
        # SampledZeroNoParityReservoirGameEnvironment,
        # SampledRollReservoirGameEnvironment,
        # SampledRoll2ReservoirGameEnvironment,
        # Sampled10ReservoirGameEnvironment,
        # Sampled10NoParityReservoirGameEnvironment,
        # SequenceReservoirGameEnvironment,
        # SampledRowReservoirGameEnvironment,
        # SampledRoll4ReservoirGameEnvironment,
        # SampledRoll41ReservoirGameEnvironment,
        # SampledRoll50ReservoirGameEnvironment,
        # SampledRoll51ReservoirGameEnvironment,
        # SampledRoll52ReservoirGameEnvironment,
        # SampledRoll53ReservoirGameEnvironment,
    ]:
        try:
            test(-1, nn, env, suffix="0", num_games=10_000,
                 print_history=False)
        except FileNotFoundError:
            pass

# tournament(MinimaxAgent(), MinimaxAgent(), num_games=100, print_history=True)
