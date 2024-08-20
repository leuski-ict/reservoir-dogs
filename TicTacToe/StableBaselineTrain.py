from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from TicTacToe import SimpleTicTacToe as TicTacToe
from gymnasium import spaces, Env
import numpy as np


class TicTacToeEnv(Env):
    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        self.current_player = 1
        self.ticTacToe = TicTacToe()
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


env = TicTacToeEnv()
check_env(env)

model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=100000)
model.save("dqn_tictactoe")
