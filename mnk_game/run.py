import tabular_policies as tp
import train
from policies import RandomPolicy, MCTSDefaultPolicy
from play import play
import tictactoe
import mnk_game
import nd_game
import ultimate


def play_random(game):
    policy = RandomPolicy()
    play(policy, game.X_MOVE, game=game)


def fit_q_and_play(game):
    policy = tp.EpsilonGreedyTabularQPolicy(epsilon=0.2)
    train.fit_q(policy, game=game, selfplay_count=500000)
    policy.epsilon = 0.
    play(policy, game.O_MOVE, game=game, verbose=True)


def policy_iteration_and_play(game):
    policy = tp.BoltzmannTabularVPolicy(temperature=0.2)
    history = train.policy_iteration(policy, game=game, selfplay_count=100000, batch_size=25, learning_rate=0.1)
    print(history)
    policy.temperature = 0.1
    play(policy, game.X_MOVE, game=game, verbose=True)


def q_policy_iteration_and_play(game):
    policy = tp.BoltzmannTabularQPolicy(temperature=0.2)
    history = train.q_policy_iteration(policy, game=game, selfplay_count=100000, batch_size=25, learning_rate=0.1)
    print(history)
    play_policy = tp.GreedyTabularQPolicy(policy.q_function)
    play(play_policy, game.X_MOVE, game=game, verbose=True)


def play_mcts(game):
    policy = MCTSDefaultPolicy(rollout_count=5000, c=1, temperature=0.1, use_visits=True)
    play(policy, game.O_MOVE, game=game, verbose=False)


def dpi_and_play(game):
    default_policy = tp.BoltzmannTabularPiPolicy()
    mcts_policy = MCTSDefaultPolicy(rollout_count=100, c=1, temperature=0.1, use_visits=True, default_policy=default_policy)
    history = train.direct_policy_iteration(mcts_policy, game=game, selfplay_count=10000, batch_size=25, learning_rate=0.1)
    print(history)
    play_policy = mcts_policy.default_policy
    play(play_policy, game.O_MOVE, game=game, verbose=True)


def play_puct_default(game):
    policy = tp.TabularPUCTDefaultPolicy(rollout_count=1000, c=1, temperature=0.1, use_visits=False)
    play(policy, game.X_MOVE, game=game, verbose=False)


def puct_v_iteration_and_play(game):
    policy = tp.TabularVTabularPUCTPolicy(rollout_count=100, c=1, temperature=1, use_visits=True)
    history = train.puct_v_iteration(policy, game, selfplay_count=5000, batch_size=25, learning_rate=0.1)
    print(history)
    policy.temperature = 0.1
    play(policy, game.X_MOVE, game=game, verbose=True)


def puct_predictor_iteration_and_play(game):
    policy = tp.TabularPUCTDefaultPolicy(rollout_count=100, c=1, temperature=1, use_visits=True)
    history = train.puct_predictor_iteration(policy, game, selfplay_count=5000, batch_size=25, learning_rate=0.1)
    print(history)
    policy.temperature = 0.1
    play(policy, game.X_MOVE, game=game, verbose=True)


def puct_v_iteration_and_play_only_predictor(game):
    policy = tp.TabularVTabularPUCTPolicy(rollout_count=100, c=1, temperature=1, use_visits=True)
    history = train.puct_v_iteration(policy, game, selfplay_count=10000, batch_size=25, learning_rate=0.1)
    print(history)
    play_policy = tp.GreedyTabularPiPolicy(pi_function=policy.pi_function)
    play(play_policy, game.O_MOVE, game=game, verbose=True)


if __name__ == '__main__':
    puct_predictor_iteration_and_play(tictactoe.TicTacToePredictor)
