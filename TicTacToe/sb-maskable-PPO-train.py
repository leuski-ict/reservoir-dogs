from TicTacToe import train_one
from Networks import *
from environments.DefaultGameEnvironment import *
from environments.ReservoirGameEnvironment import *

# train_one(-1, DefaultNN, DefaultGameEnvironment)
train_one(-1, DefaultNN, DefaultGameEnvironment, suffix="fast-reward-1")

# for nn in [
#     # SimpleNN,
#     # OneLayerNN,
#     # OneLayerWithTanhNN,
#     # TwoLayerNN,
#     # TwoLayerWithTanhSoftmaxNN,
#     # TwoLayerRReluNN,
#     BitDecoder1NN,
# ]:
#     for env in [
#         # MeanReservoirGameEnvironment,
#         # MeanNoParityReservoirGameEnvironment,
#         SampledReservoirGameEnvironment,
#         SampledNoParityReservoirGameEnvironment,
#         # DecodingReservoirGameEnvironment,
#     ]:
#         train_one(-1, nn, env, steps=1_000_000)
