import policies
import train
from games import TicTacToe
from play import play


def play_random():
    policy = policies.RandomPolicy()
    play(policy, TicTacToe.X_MOVE, game=TicTacToe)


def fit_q_and_play():
    policy = policies.EpsilonGreedyTabularQPolicy(epsilon=0.2)
    train.fit_q(policy, game=TicTacToe, num_rollouts=10000)
    policy.epsilon = 0.
    play(policy, TicTacToe.X_MOVE, game=TicTacToe, verbose=True)


def policy_iteration_and_play():
    policy = policies.BoltzmannTabularVPolicy(temperature=0.2)
    train.policy_iteration(policy, game=TicTacToe, num_rollouts=50000, batch_size=50, learning_rate=0.01)
    policy.temperature = 0.1
    play(policy, TicTacToe.X_MOVE, game=TicTacToe, verbose=True)


def q_policy_iteration_and_play():
    policy = policies.BoltzmannTabularQPolicy(temperature=0.2)
    train.q_policy_iteration(policy, game=TicTacToe, num_rollouts=50000, batch_size=50, learning_rate=0.01)
    play_policy = policies.GreedyTabularQPolicy(policy.q_function)
    play(play_policy, TicTacToe.X_MOVE, game=TicTacToe, verbose=True)


if __name__ == '__main__':
    policy_iteration_and_play()
