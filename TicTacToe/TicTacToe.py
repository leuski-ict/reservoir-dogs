import random
import os
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO

from Board import *
from agents.MinimaxAgent import MinimaxAgent
from agents.StableBaselinesMaskableAgent import StableBaselineMaskableAgent
from environments.MaskableStableBaselineEnvironment import \
    MaskableStableBaselineEnvironment
from environments.ReservoirGameEnvironment import *
from environments.StableBaselineEnvironment import check_env

# RESERVOIR_ENV = MeanReservoirGameEnvironment
RESERVOIR_ENV = SampledReservoirGameEnvironment


def evaluate(env, action_fn, num_games=100):
    print("starting evaluation")
    wins = {1: 0, -1: 0, 0: 0, None: 0}
    for _ in range(num_games):
        obs = env.reset()[0]
        history = BoardList()
        while not env.tic_tac_toe.done:
            action = action_fn(obs)
            obs = env.step(action)[0]
            history.append(env.tic_tac_toe.board.copy())
            if env.tic_tac_toe.done:
                print(history)
                print("winner", env.tic_tac_toe.winner)
                wins[env.tic_tac_toe.winner] += 1
    print("Wins:", wins)
    return wins[0]


def tournament(x_agent, o_agent, num_games=100):
    print("starting tournament")
    wins = {1: 0, -1: 0, 0: 0, None: 0}
    game = Game()
    for _ in range(num_games):
        game.reset()
        action = random.choice(game.available_actions())
        game.make_move(action, 1)

        history = BoardList()
        while not game.done:
            (x_agent if game.current_player == 1 else o_agent).make_move(game)
            history.append(game.board.copy())
            if game.done:
                print(history)
                print("winner", game.winner)
                wins[game.winner] += 1
    print("Wins:", wins)
    return wins[0]


def test(player, suffix):
    file_name = StableBaselineMaskableAgent.file_name(player, suffix)
    test_agent = StableBaselineMaskableAgent(
        MaskablePPO.load(file_name), RESERVOIR_ENV)
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


def opponent(player, suffix):
    try:
        model = MaskablePPO.load(
            StableBaselineMaskableAgent.file_name(player, suffix))
        return StableBaselineMaskableAgent(model, RESERVOIR_ENV)
    except ValueError:
        return MinimaxAgent()


def get_model(player, suffix, nn_type):
    log_dir = "../tmp/"
    os.makedirs(log_dir, exist_ok=True)
    game_env = RESERVOIR_ENV()
    our_env = MaskableStableBaselineEnvironment(game_env, player)
    env = check_env(Monitor(our_env, log_dir))
    #    callback = EvalCallback(eval_freq=1000, log_path=log_dir)
    try:
        return MaskablePPO.load(
            StableBaselineMaskableAgent.file_name(player, suffix),
            env=env), our_env
    except FileNotFoundError:
        policy_kwargs = dict(
            features_extractor_class=nn_type,
            features_extractor_kwargs=dict(features_dim=game_env.output_count)
        )
        return MaskablePPO('MlpPolicy', env, policy_kwargs=policy_kwargs,
                           verbose=1, tensorboard_log=log_dir), our_env


def train_one(player, nn_type, suffix=None):
    if suffix is None:
        suffix = nn_type.__name__
    model, env = get_model(player, suffix, nn_type)
    env.opponent = MinimaxAgent()
    model.learn(total_timesteps=1000000)
    model.save(StableBaselineMaskableAgent.file_name(player, suffix))


def train_multiple(suffix, nn_type):
    model_x, env_x = get_model(1, suffix, nn_type)
    model_o, env_o = get_model(-1, suffix, nn_type)
    env_x.opponent = StableBaselineMaskableAgent(
        model_o, RESERVOIR_ENV)
    env_o.opponent = StableBaselineMaskableAgent(
        model_x, RESERVOIR_ENV)
    for iteration in range(10000):
        model_x.learn(total_timesteps=100)
        model_o.learn(total_timesteps=100)
        if iteration % 100 == 0:
            model_x.save(StableBaselineMaskableAgent.file_name(1, suffix))
            model_o.save(StableBaselineMaskableAgent.file_name(-1, suffix))
    model_x.save(StableBaselineMaskableAgent.file_name(1, suffix))
    model_o.save(StableBaselineMaskableAgent.file_name(-1, suffix))
