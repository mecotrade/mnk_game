from colorama import Fore, Style

from contexts import Context


class TicTacToe(Context):

    X_MOVE = 1
    O_MOVE = -1
    NUM_ACTIONS = 9
    WIN_POSITIONS = [
        0b_000_000_111,
        0b_000_111_000,
        0b_111_000_000,
        0b_001_001_001,
        0b_010_010_010,
        0b_100_100_100,
        0b_100_010_001,
        0b_001_010_100
    ]

    @staticmethod
    def new():
        return TicTacToe((0, 0))

    @staticmethod
    def calculate_reward(board):
        for position in TicTacToe.WIN_POSITIONS:
            if board & position == position:
                return 1
        else:
            return 0

    @staticmethod
    def to_bits(state):
        bits = []
        for _ in range(TicTacToe.NUM_ACTIONS):
            bits = bits + [state % 2]
            state = state // 2
        return bits

    def analyze(self):
        board_x, board_o = self.board
        reward_x = TicTacToe.calculate_reward(board_x)
        reward_o = -TicTacToe.calculate_reward(board_o)

        bits_x = TicTacToe.to_bits(board_x)
        bits_o = TicTacToe.to_bits(board_o)
        nx = sum(bits_x)
        no = sum(bits_o)
        if nx == no and reward_x == 0:
            move = TicTacToe.X_MOVE
            reward = reward_o
        elif nx == no + 1 and reward_o == 0:
            move = TicTacToe.O_MOVE
            reward = reward_x
        else:
            move = None
            reward = 0

        board_free = TicTacToe.to_bits(~(board_x | board_o))

        done = reward != 0 or sum(board_free) == 0
        actions = []
        for i in range(TicTacToe.NUM_ACTIONS):
            if board_free[i] == 1:
                actions = actions + [i]

        return reward, done, move, actions

    def apply(self, action):
        board_x, board_o = self.board
        if self.move == TicTacToe.X_MOVE:
            board_x += 2 ** action
        else:
            board_o += 2 ** action
        return board_x, board_o

    def render(self):

        def cell(x, o, pos):
            if x == 1:
                return Fore.RED + ' X ' + Style.RESET_ALL
            elif o == 1:
                return Fore.BLUE + ' O ' + Style.RESET_ALL
            else:
                return Fore.YELLOW + f' {pos+1} ' + Style.RESET_ALL

        board_x, board_o = self.board
        bits_x = TicTacToe.to_bits(board_x)
        bits_o = TicTacToe.to_bits(board_o)
        cells = [cell(bits_x, bits_o, pos) for pos, (bits_x, bits_o) in enumerate(zip(bits_x, bits_o))]

        border_line = '+'.join(['---'] * 3)
        print(f'+{border_line}+')
        for row in range(0, TicTacToe.NUM_ACTIONS, 3):
            table_row = '|'.join(cells[row:row + 3])
            print(f'|{table_row}|')
            print(f'+{border_line}+')

        if not self.done:
            if self.move == TicTacToe.X_MOVE:
                print('Crosses move')
            elif self.move == TicTacToe.O_MOVE:
                print('Noughts move')
        else:
            if self.reward == 1:
                print('Crosses win')
            elif self.reward == -1:
                print('Noughts win')
            else:
                print('Draw')
