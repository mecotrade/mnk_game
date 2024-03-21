from typing import Type

import tabular_policies
import net_policies
import tabular_train
import net_train
from contexts import Context, ContextPredictor
from policies import RandomPolicy, MCTSDefaultPolicy
from play import play
import tictactoe
import mnk_game
import nd_game
import ultimate
import models


def play_random(game):
    policy = RandomPolicy()
    play(policy, game.X_MOVE, game=game)


def fit_q_and_play(game):
    policy = tabular_policies.EpsilonGreedyTabularQPolicy(epsilon=0.2)
    tabular_train.fit_q(policy, game=game, num_games=500000)
    policy.epsilon = 0.
    play(policy, game.O_MOVE, game=game, verbose=True)


def policy_iteration_and_play(game):
    policy = tabular_policies.BoltzmannTabularVPolicy(temperature=0.2)
    history = tabular_train.policy_iteration(policy, game=game, num_games=100000, batch_size=25, learning_rate=0.1)
    print(history)
    policy.temperature = 0.1
    play(policy, game.X_MOVE, game=game, verbose=True)


def q_policy_iteration_and_play(game):
    policy = tabular_policies.BoltzmannTabularQPolicy(temperature=0.2)
    history = tabular_train.q_policy_iteration(policy, game=game, num_games=100000, batch_size=25, learning_rate=0.1)
    print(history)
    play_policy = tabular_policies.GreedyTabularQPolicy(policy.q_function)
    play(play_policy, game.X_MOVE, game=game, verbose=True)


def play_mcts(game):
    policy = MCTSDefaultPolicy(rollout_count=5000, c=1, temperature=0.1, use_visits=True)
    play(policy, game.O_MOVE, game=game, verbose=False)


def dpi_and_play(game):
    default_policy = tabular_policies.BoltzmannTabularPiPolicy()
    mcts_policy = MCTSDefaultPolicy(rollout_count=100, c=1, temperature=0.1, use_visits=True, default_policy=default_policy)
    history = tabular_train.direct_policy_iteration(mcts_policy, game=game, num_games=10000, batch_size=25, learning_rate=0.1)
    print(history)
    play_policy = mcts_policy.default_policy
    play(play_policy, game.O_MOVE, game=game, verbose=True)


def play_puct_default(game):
    policy = tabular_policies.TabularPUCTDefaultPolicy(rollout_count=1000, c=1, temperature=0.1, use_visits=False)
    play(policy, game.X_MOVE, game=game, verbose=False)


def puct_v_iteration_and_play(game):
    policy = tabular_policies.TabularVTabularPUCTPolicy(rollout_count=100, c=1, temperature=1, use_visits=True)
    history = tabular_train.puct_v_iteration(policy, game, num_games=5000, batch_size=25, learning_rate=0.1)
    print(history)
    policy.temperature = 0.1
    play(policy, game.X_MOVE, game=game, verbose=True)


def puct_predictor_iteration_and_play(game):
    policy = tabular_policies.TabularPUCTDefaultPolicy(rollout_count=100, c=1, temperature=1, use_visits=True)
    history = tabular_train.puct_predictor_iteration(policy, game, num_games=5000, batch_size=25, learning_rate=0.1)
    print(history)
    policy.temperature = 0.1
    play(policy, game.X_MOVE, game=game, verbose=True)


def puct_v_iteration_and_play_only_predictor(game):
    policy = tabular_policies.TabularVTabularPUCTPolicy(rollout_count=100, c=1, temperature=1, use_visits=True)
    history = tabular_train.puct_v_iteration(policy, game, num_games=10000, batch_size=25, learning_rate=0.1)
    print(history)
    play_policy = tabular_policies.GreedyTabularPiPolicy(pi_function=policy.pi_function)
    play(play_policy, game.O_MOVE, game=game, verbose=True)


def net_policy_iteration_value_best_response(game):
    model = models.VModel(game.shape())
    policy = net_policies.EpsilonGreedyNetVPolicy(model, epsilon=0.1)
    opponent = RandomPolicy()
    net_train.policy_iteration(policy, opponent, game, 100000, 5, 25, 50,
                               2500, 1e-3, 1000)


def net_policy_iteration(game, num_blocks=4):
    run_id = None
    num_games_start = 0
    model = models.VModel(game.shape(), num_blocks=num_blocks)
    policy = net_policies.EpsilonGreedyNetVPolicy(model, epsilon=0.1)
    opponent_model = models.VModel(game.shape(), num_blocks=num_blocks)
    opponent_model.load_state_dict(model.state_dict())
    opponent = net_policies.EpsilonGreedyNetVPolicy(opponent_model, epsilon=0.1)

    net_train.policy_iteration(policy, opponent, game, 100000, 5, 25, 50,
                               2500, 1e-3, 1000, 0.55, run_id, num_games_start)


def net_policy_iteration_selfplay(game):
    run_id = None
    num_games_start = 0
    model = models.VModel(game.shape())
    policy = net_policies.EpsilonGreedyNetVPolicy(model, epsilon=0.1)

    net_train.policy_iteration_selfplay(policy, game, 100000, 5, 25, 50,
                                        2500, 1e-3, 1000, run_id, num_games_start)


def net_policy_iteration_q(game: Type[Context], num_blocks=2):
    run_id = None
    num_games_start = 0
    model = models.QModel(game.shape(), game.num_actions(), num_blocks=num_blocks)
    policy = net_policies.EpsilonGreedyNetQPolicy(model, epsilon=0.1)
    opponent_model = models.QModel(game.shape(), game.num_actions(), num_blocks=num_blocks)
    opponent_model.load_state_dict(model.state_dict())
    opponent = net_policies.EpsilonGreedyNetQPolicy(opponent_model, epsilon=0.1)

    net_train.q_policy_iteration(policy, opponent, game, 100000, 5, 25, 50,
                                 2500, 1e-3, 1000, 0.55, run_id, num_games_start)


def net_puct_predictor(game: Type[ContextPredictor]):
    default_policy = RandomPolicy()
    model = models.PredictorModel(game.shape(), game.num_actions())
    policy = net_policies.NetPUCTDefaultPolicy(model, 25, use_visits=True, temperature=0.1, default_policy=default_policy)
    net_train.puct_predictor_iteration(policy, game, 100000, 5, 1e-3, 500)


def train_alpha(game):
    model = models.AlphaModel(game.shape(), game.num_actions())
    policy = net_policies.AlphaPolicy(model, rollout_count=25, c=1, temperature=1, use_visits=True)
    opponent_model = models.AlphaModel(game.shape(), game.num_actions())
    opponent_model.load_state_dict(model.state_dict())
    opponent = net_policies.AlphaPolicy(opponent_model, rollout_count=25, c=1, temperature=1, use_visits=True)
    net_train.alpha_iteration(policy, opponent, game, 100000, 5, 25, 50, 2500, 1e-3, 500)


if __name__ == '__main__':
    net_policy_iteration(ultimate.UltimateTicTacToe, num_blocks=1)
