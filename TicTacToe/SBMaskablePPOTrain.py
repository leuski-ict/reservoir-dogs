from typing import Optional
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from TicTacToe import SimpleTicTacToe as TicTacToe
from gymnasium import spaces
import numpy as np


# noinspection DuplicatedCode
class TicTacToeEnv(InvalidActionEnvDiscrete):
    def __init__(
            self,
            dim: Optional[int] = None,
            ep_length: int = 100,
            n_invalid_actions: int = 0):
        self.current_player = 1
        self.ticTacToe = TicTacToe()
        super(TicTacToeEnv, self).__init__(dim, ep_length, n_invalid_actions)
        self.action_space = spaces.Discrete(self.ticTacToe.output_count)
        self.observation_space = spaces.Box(
            low=self.ticTacToe.min_input_value,
            high=self.ticTacToe.max_input_value,
            shape=(self.ticTacToe.input_count,), dtype=np.float32)
        self.reset()

    def reset(self, **kwargs):
        state, _ = self.ticTacToe.reset()
        self.current_player = self.ticTacToe.current_player
        return state, {}

    def step(self, action):
        state, done, reward = self.ticTacToe.step(
            self.current_player, divmod(action, self.ticTacToe.board_size))
        self.current_player *= -1
        return state, reward, done, False, {}

    def render(self, mode='human'):
        return

    def action_masks(self):
        result = [x == 0 for x in self.ticTacToe.board.flatten()]
        return result


env = TicTacToeEnv()
check_env(env)

model = MaskablePPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=100000)
evaluate_policy(model, env, n_eval_episodes=20, warn=False)

model.save("maskable_ppo_tictactoe")
