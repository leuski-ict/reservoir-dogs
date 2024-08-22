import random
from Game import *


def board_history_as_string(boards: [Board]) -> str:
    return "\n".join([" ".join([board.row_as_string(row)
                                for board in boards])
                      for row in range(boards[0].size)])


def evaluate(env, action_fn, num_games=100):
    print("starting evaluation")
    wins = {1: 0, -1: 0, 0: 0, None: 0}
    for _ in range(num_games):
        obs = env.reset()[0]
        history = []
        while not env.tic_tac_toe.done:
            action = action_fn(obs)
            obs = env.step(action)[0]
            history.append(env.tic_tac_toe.board.copy())
            if env.tic_tac_toe.done:
                print("winner", env.tic_tac_toe.winner)
                print(board_history_as_string(history))
                wins[env.tic_tac_toe.winner] += 1
    print("Wins:", wins)


def tournament(x_agent, o_agent, num_games=100):
    print("starting tournament")
    wins = {1: 0, -1: 0, 0: 0, None: 0}
    game = Game()
    for _ in range(num_games):
        game.reset()
        action = random.choice(game.available_actions())
        game.make_move(action, 1)

        history = []
        while not game.done:
            (x_agent if game.current_player == 1 else o_agent).make_move(game)
            history.append(game.board.copy())
            if game.done:
                print("winner", game.winner)
                print(board_history_as_string(history))
                wins[game.winner] += 1
    print("Wins:", wins)
