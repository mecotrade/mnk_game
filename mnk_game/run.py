import policies
import train
from games import TicTacToe, TicTacToeTree, MNKGame544Tree, MNKGame554Tree, MNKGame333Tree
from play import play


def play_random():
    policy = policies.RandomPolicy()
    play(policy, TicTacToe.X_MOVE, game=TicTacToe)


def fit_q_and_play():
    policy = policies.EpsilonGreedyTabularQPolicy(epsilon=0.2)
    train.fit_q(policy, game=TicTacToe, selfplay_count=25000)
    policy.epsilon = 0.
    play(policy, TicTacToe.X_MOVE, game=TicTacToe, verbose=True)


def policy_iteration_and_play():
    policy = policies.BoltzmannTabularVPolicy(temperature=0.2)
    history = train.policy_iteration(policy, game=TicTacToe, selfplay_count=100000, batch_size=25, learning_rate=0.1)
    print(history)
    policy.temperature = 0.1
    play(policy, TicTacToe.X_MOVE, game=TicTacToe, verbose=True)


def q_policy_iteration_and_play():
    policy = policies.BoltzmannTabularQPolicy(temperature=0.2)
    history = train.q_policy_iteration(policy, game=TicTacToe, selfplay_count=100000, batch_size=25, learning_rate=0.1)
    print(history)
    play_policy = policies.GreedyTabularQPolicy(policy.q_function)
    play(play_policy, TicTacToe.X_MOVE, game=TicTacToe, verbose=True)


def play_mcts():
    policy = policies.MCTSDefaultPolicy(rollout_count=100, c=1, temperature=0.1, use_visits=False)
    play(policy, TicTacToeTree.X_MOVE, game=TicTacToeTree, verbose=True)


def play_mcts_mnk544():
    policy = policies.MCTSDefaultPolicy(rollout_count=10000, c=1, temperature=0.1, use_visits=True)
    play(policy, MNKGame544Tree.X_MOVE, game=MNKGame544Tree, verbose=True)


def play_mcts_mnk554():
    policy = policies.MCTSDefaultPolicy(rollout_count=10000, c=1, temperature=0.1, use_visits=True)
    play(policy, MNKGame544Tree.O_MOVE, game=MNKGame554Tree, verbose=True)


def dpi_and_play():
    default_policy = policies.BoltzmannTabularPiPolicy()
    mcts_policy = policies.MCTSDefaultPolicy(rollout_count=100, c=1, temperature=0.1, use_visits=True, default_policy=default_policy)
    history = train.direct_policy_iteration(mcts_policy, game=TicTacToeTree, selfplay_count=10000, batch_size=25, learning_rate=0.1)
    print(history)
    play_policy = mcts_policy.default_policy
    play(play_policy, TicTacToeTree.O_MOVE, game=TicTacToeTree, verbose=True)


def puct_and_play():
    policy = policies.TabularPUCTPolicy(rollout_count=100, c=1, temperature=0.1, use_visits=False)
    history = train.puct(policy, TicTacToeTree, selfplay_count=5000, batch_size=25, learning_rate=0.1)
    print(history)
    policy.temperature = 0.1
    play(policy, TicTacToeTree.O_MOVE, game=TicTacToeTree, verbose=True)


def puct_and_play333():

    policy = policies.TabularPUCTPolicy(rollout_count=100, c=1, temperature=0.1, use_visits=False)
    history = train.puct(policy, MNKGame333Tree, selfplay_count=5000, batch_size=25, learning_rate=0.1)
    print(history)
    policy.temperature = 0.1
    play(policy, MNKGame333Tree.O_MOVE, game=MNKGame333Tree, verbose=True)


def puct_and_play_only_pi():
    policy = policies.TabularPUCTPolicy(rollout_count=100, c=1, temperature=1, use_visits=True)
    history = train.puct(policy, TicTacToeTree, selfplay_count=5000, batch_size=25, learning_rate=0.1)
    print(history)
    play_policy = policies.GreedyTabularPiPolicy(pi_function=policy.pi_function)
    play(play_policy, TicTacToeTree.O_MOVE, game=TicTacToeTree, verbose=True)


if __name__ == '__main__':
    play_mcts_mnk554()
