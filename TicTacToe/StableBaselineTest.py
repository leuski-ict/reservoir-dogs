from stable_baselines3 import DQN, PPO
from StableBaselineTrain import TicTacToeEnv


def evaluate_model(model, env, num_games=100):
    wins = {1: 0, -1: 0, 0: 0, None: 0}
    for _ in range(num_games):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            if done:
                print("winner", env.ticTacToe.winner)
                print(env.ticTacToe.board)
                wins[env.ticTacToe.winner] += 1
    print("Wins:", wins)


environment = TicTacToeEnv()
# Load the trained model
sb_model = PPO.load("dqn_tictactoe")
evaluate_model(sb_model, environment)
