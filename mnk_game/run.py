from policies import RandomPolicy, EpsilonGreedyTabularQPolicy
from games import TicTacToe
from play import play
from train import fit_q


def play_random():
    policy = RandomPolicy()
    play(policy, TicTacToe.X_MOVE, game=TicTacToe)


def fit_q_and_play():
    policy = EpsilonGreedyTabularQPolicy(epsilon=0.2)
    fit_q(policy, game=TicTacToe, num_rollouts=10000)
    policy.epsilon = 0.
    play(policy, TicTacToe.X_MOVE, game=TicTacToe, verbose=True)


if __name__ == '__main__':
    fit_q_and_play()
