from sb3_contrib import MaskablePPO

from TicTacToe import *
from agents.MinimaxAgent import TicTacToeMinimaxAgent
from agents.StableBaselinesMaskableAgent import SBTicTacToeMaskableAgent
from environments.MemristorGameEnvironment import TicTacToeFloatBits


def test(player, suffix):
    file_name = SBTicTacToeMaskableAgent.file_name(player, suffix)
    test_agent = SBTicTacToeMaskableAgent(
        MaskablePPO.load(file_name), TicTacToeFloatBits)
    if player == 1:
        tournament(
            test_agent,
            TicTacToeMinimaxAgent(),
        )
    else:
        tournament(
            TicTacToeMinimaxAgent(),
            test_agent,
        )


test(-1, "simple")