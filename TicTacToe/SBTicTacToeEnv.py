from TicTacToe import TicTacToe
from gymnasium import spaces, Env
import numpy as np
import stable_baselines3.common.env_checker


class SBTicTacToeEnv(Env):
    def __init__(self, tic_tac_toe: TicTacToe):
        self.current_player = 1
        self.ticTacToe = tic_tac_toe
        Env.__init__(self)
        self.action_space = self.make_action_space()
        self.observation_space = self.make_observation_space()
        self.reset()

    def make_action_space(self):
        return spaces.Discrete(self.ticTacToe.output_count)

    def make_observation_space(self):
        return spaces.Box(
            low=self.ticTacToe.min_input_value,
            high=self.ticTacToe.max_input_value,
            shape=(self.ticTacToe.input_count,), dtype=np.float32)

    def reset(self, **kwargs):
        state, _ = self.ticTacToe.reset()
        self.current_player = self.ticTacToe.current_player
        return state, {}

    def step(self, action):
        state, done, reward = self.ticTacToe.step(
            action=action, player=self.current_player)
        self.current_player *= -1
        return state, reward, done, False, {}

    def render(self, mode='human'):
        return

    def action_masks(self):
        result = [x == 0 for x in self.ticTacToe.board.flatten()]
        return result


def check_env(env):
    stable_baselines3.common.env_checker.check_env(env)
    return env
