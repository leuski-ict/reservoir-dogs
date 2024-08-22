from sb3_contrib import MaskablePPO

from TicTacToe import *
from agents.MinimaxAgent import MinimaxAgent
from agents.StableBaselinesMaskableAgent import StableBaselineMaskableAgent
from environments.ReservoirGameEnvironment import MeanReservoirGameEnvironment


def test(player, suffix):
    file_name = StableBaselineMaskableAgent.file_name(player, suffix)
    test_agent = StableBaselineMaskableAgent(
        MaskablePPO.load(file_name), MeanReservoirGameEnvironment)
    if player == 1:
        tournament(
            test_agent,
            MinimaxAgent(),
        )
    else:
        tournament(
            MinimaxAgent(),
            test_agent,
        )


test(-1, "simple")