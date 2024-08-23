from TicTacToe import *
from Networks import *
from environments.DefaultGameEnvironment import *
from environments.ReservoirGameEnvironment import *

test(-1, DefaultNN, DefaultGameEnvironment, suffix="fast-reward-1",
     num_games=10000,
     print_history=False)

# for nn in [
#     # SimpleNN,
#     # OneLayerNN,
#     # OneLayerWithTanhNN,
#     # TwoLayerNN,
#     # TwoLayerWithTanhSoftmaxNN,
#     TwoLayerReluNN,
#     TwoLayerRReluNN,
# ]:
#     for env in [
#         # MeanReservoirGameEnvironment,
#         # MeanNoParityReservoirGameEnvironment,
#         SampledReservoirGameEnvironment,
#         SampledNoParityReservoirGameEnvironment
#     ]:
#         test(-1, nn, env, num_games=10000,
#              print_history=False)

# tournament(MinimaxAgent(), MinimaxAgent(), num_games=100, print_history=True)
