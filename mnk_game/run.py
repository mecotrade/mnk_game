from policies import RandomPolicy
from games import TicTacToe
from play import play


def play_random():
    policy = RandomPolicy()
    play(policy, TicTacToe.X_MOVE, game=TicTacToe)


if __name__ == '__main__':
    play_random()
