from gymnasium import spaces, Env
import numpy as np
import stable_baselines3.common.env_checker

from agents.Agent import AbstractAgent
from environments.GameEnvironment import GameEnvironment


class StableBaselineEnvironment(Env):
    def __init__(self, tic_tac_toe: GameEnvironment, player,
                 opponent: AbstractAgent = None):
        self.this_player = player
        self.tic_tac_toe = tic_tac_toe
        self.opponent = opponent
        Env.__init__(self)
        self.action_space = self.make_action_space()
        self.observation_space = self.make_observation_space()
        self.reset()

    def make_action_space(self):
        return spaces.Discrete(self.tic_tac_toe.output_count)

    def make_observation_space(self):
        return spaces.Box(
            low=self.tic_tac_toe.min_input_value,
            high=self.tic_tac_toe.max_input_value,
            shape=(self.tic_tac_toe.input_count,), dtype=np.float32)

    def reset(self, **kwargs):
        self.tic_tac_toe.reset()
        if self.opponent is not None:
            while self.tic_tac_toe.game.current_player != self.this_player:
                self.opponent.make_move(self.tic_tac_toe.game)
        return self.tic_tac_toe.observation(self.this_player), {}

    def step(self, action):
        state, done, reward = self.tic_tac_toe.step(
            action=action, player=self.this_player)
        if not done and self.opponent is not None:
            self.opponent.make_move(self.tic_tac_toe.game)
            # you can change this that it uses the minimaxAgent estimation
            # to return the expected reward immediately. However, in our
            # experiments a default agent trained that way performs worse.
            state, done, reward = self.tic_tac_toe.step_return(self.this_player)
        return state, reward, done, False, {}

    def render(self, mode='human'):
        return

    def action_masks(self):
        return self.tic_tac_toe.action_masks()


def check_env(env):
    stable_baselines3.common.env_checker.check_env(env)
    return env
