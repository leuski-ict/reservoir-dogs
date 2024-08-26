import random
import os
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO

from Board import *
from Game import Game
from agents.MinimaxAgent import MinimaxAgent
from agents.StableBaselinesMaskableAgent import StableBaselineMaskableAgent
from environments.MaskableStableBaselineEnvironment import \
    MaskableStableBaselineEnvironment
from environments.StableBaselineEnvironment import check_env

# torch device. "mps" will use the GPU, but in my experiments
# this is way, way, way slower than using cpu. Most likely because of
# small matrices involved.
DEVICE = "cpu"


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


def tournament(x_agent, o_agent, name="", num_games=100, print_history=False):
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
                if print_history:
                    print(history)
                    print("winner", game.winner)
                wins[game.winner] += 1
    print(name, "Wins:", wins)
    return wins[0]


def test(player, nn_type, game_env_type, suffix=None,
         num_games=100, print_history=False):
    name = experiment_name(locals())
    file_name = StableBaselineMaskableAgent.file_name(name)
    test_agent = StableBaselineMaskableAgent(
        MaskablePPO.load(file_name), game_env_type)
    if player == 1:
        tournament(
            test_agent,
            MinimaxAgent(),
            name=name,
            num_games=num_games,
            print_history=print_history,
        )
    else:
        tournament(
            MinimaxAgent(),
            test_agent,
            name=name,
            num_games=num_games,
            print_history=print_history,
        )


def experiment_name(args):
    result = []
    if "player" in args:
        result.append("x" if args["player"] == 1 else "o")
    if "game_env_type" in args:
        game_env_type = args["game_env_type"]
        result.append(
            game_env_type.name if hasattr(game_env_type,
                                          "name") else game_env_type.__name__)
    if "nn_type" in args:
        nn_type = args["nn_type"]
        result.append(
            nn_type.name if hasattr(nn_type, "name") else nn_type.__name__)
    if "suffix" in args:
        suffix = args["suffix"]
        if suffix is not None and len(suffix) > 0:
            result.append(suffix)
    return "_".join(result)


def opponent(player, nn_type, game_env_type, suffix=None):
    name = experiment_name(locals())
    log_dir = "../tmp/"
    os.makedirs(log_dir, exist_ok=True)
    try:
        model = MaskablePPO.load(
            StableBaselineMaskableAgent.file_name(name))
        return StableBaselineMaskableAgent(model, game_env_type)
    except ValueError:
        return MinimaxAgent()


def get_model(player, nn_type, game_env_type, suffix=None):
    name = experiment_name(locals())
    log_dir = "../tmp/"
    os.makedirs(log_dir, exist_ok=True)
    game_env = game_env_type()
    our_env = MaskableStableBaselineEnvironment(game_env, player)
    env = check_env(Monitor(our_env, log_dir))
    #    callback = EvalCallback(eval_freq=1000, log_path=log_dir)
    try:
        return MaskablePPO.load(
            StableBaselineMaskableAgent.file_name(name),
            device=DEVICE,
            env=env), our_env
    except FileNotFoundError:
        policy_kwargs = dict(
            features_extractor_class=nn_type,
            features_extractor_kwargs=dict(
                features_dim=game_env.output_count)
        )
        return MaskablePPO('MlpPolicy', env, policy_kwargs=policy_kwargs,
                           verbose=1, device=DEVICE,
                           tensorboard_log=log_dir), our_env


def train_one(player, nn_type, game_env_type, suffix=None, steps=1_000_000):
    name = experiment_name(locals())
    model, env = get_model(player, nn_type, game_env_type, suffix=suffix)
    env.opponent = MinimaxAgent()
    model.learn(total_timesteps=steps, tb_log_name=name)
    model.save(StableBaselineMaskableAgent.file_name(name))


def train_multiple(nn_type, game_env_type, suffix=None):
    name_values = locals()
    x_file_name = StableBaselineMaskableAgent.file_name(
        experiment_name({"player": 1} | name_values))
    o_file_name = StableBaselineMaskableAgent.file_name(
        experiment_name({"player": 1} | name_values))

    model_x, env_x = get_model(1, nn_type, game_env_type, suffix=suffix)
    model_o, env_o = get_model(-1, nn_type, game_env_type, suffix=suffix)
    env_x.opponent = StableBaselineMaskableAgent(
        model_o, game_env_type)
    env_o.opponent = StableBaselineMaskableAgent(
        model_x, game_env_type)
    for iteration in range(10_000):
        model_x.learn(total_timesteps=100)
        model_o.learn(total_timesteps=100)
        if iteration % 100 == 0:
            model_x.save(x_file_name)
            model_o.save(o_file_name)
    model_x.save(x_file_name)
    model_o.save(o_file_name)
