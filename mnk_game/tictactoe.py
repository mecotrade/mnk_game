from colorama import Fore, Style

from contexts import Context, ContextTree


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

    @classmethod
    def new(cls):
        return cls((0, 0))

    @classmethod
    def num_actions(cls):
        return cls.NUM_ACTIONS

    @classmethod
    def calculate_reward(cls, board):
        for position in cls.WIN_POSITIONS:
            if board & position == position:
                return 1
        else:
            return 0

    @classmethod
    def to_bits(cls, state):
        bits = []
        for _ in range(cls.NUM_ACTIONS):
            bits = bits + [state % 2]
            state = state // 2
        return bits

    def analyze(self):
        board_x, board_o = self.board
        reward_x = self.calculate_reward(board_x)
        reward_o = -self.calculate_reward(board_o)

        bits_x = self.to_bits(board_x)
        bits_o = self.to_bits(board_o)
        x_count = sum(bits_x)
        o_count = sum(bits_o)
        if x_count == o_count and reward_x == 0:
            move = self.X_MOVE
            reward = reward_o
        elif x_count == o_count + 1 and reward_o == 0:
            move = self.O_MOVE
            reward = reward_x
        else:
            move = None
            reward = 0

        board_free = self.to_bits(~(board_x | board_o))

        done = reward != 0 or sum(board_free) == 0
        actions = []
        for i in range(self.NUM_ACTIONS):
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

    @staticmethod
    def print_board(cells):
        border_line = '+'.join(['---'] * 3)
        print(f'+{border_line}+')
        for row in range(0, TicTacToe.NUM_ACTIONS, 3):
            table_row = '|'.join(cells[row:row + 3])
            print(f'|{table_row}|')
            print(f'+{border_line}+')

    def render(self):

        def cell(x, o, pos):
            if x == 1:
                return Fore.RED + ' X ' + Style.RESET_ALL
            elif o == 1:
                return Fore.CYAN + ' O ' + Style.RESET_ALL
            else:
                return Fore.LIGHTBLACK_EX + f'{pos+1:^3}' + Style.RESET_ALL

        board_x, board_o = self.board
        bits_x = self.to_bits(board_x)
        bits_o = self.to_bits(board_o)
        cells = [cell(bits_x, bits_o, pos) for pos, (bits_x, bits_o) in enumerate(zip(bits_x, bits_o))]

        self.print_board(cells)

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


class TicTacToeTree(ContextTree, TicTacToe):
    pass


class TicTacToe3D(TicTacToe):

    NUM_ACTIONS = 27

    WIN_POSITIONS = [
        # level 1
        0b_000000000_000000000_000000111,
        0b_000000000_000000000_000111000,
        0b_000000000_000000000_111000000,
        0b_000000000_000000000_001001001,
        0b_000000000_000000000_010010010,
        0b_000000000_000000000_100100100,
        0b_000000000_000000000_100010001,
        0b_000000000_000000000_001010100,
        # level 2
        0b_000000000_000000111_000000000,
        0b_000000000_000111000_000000000,
        0b_000000000_111000000_000000000,
        0b_000000000_001001001_000000000,
        0b_000000000_010010010_000000000,
        0b_000000000_100100100_000000000,
        0b_000000000_100010001_000000000,
        0b_000000000_001010100_000000000,
        # level 3
        0b_000000111_000000000_000000000,
        0b_000111000_000000000_000000000,
        0b_111000000_000000000_000000000,
        0b_001001001_000000000_000000000,
        0b_010010010_000000000_000000000,
        0b_100100100_000000000_000000000,
        0b_100010001_000000000_000000000,
        0b_001010100_000000000_000000000,
        # vertical
        0b_000000001_000000001_000000001,
        0b_000000010_000000010_000000010,
        0b_000000100_000000100_000000100,
        0b_000001000_000001000_000001000,
        0b_000010000_000010000_000010000,
        0b_000100000_000100000_000100000,
        0b_001000000_001000000_001000000,
        0b_010000000_010000000_010000000,
        0b_100000000_100000000_100000000,
        # xz-diagonal
        0b_000000100_000000010_000000001,
        0b_000100000_000010000_000001000,
        0b_100000000_010000000_001000000,
        0b_000000001_000000010_000000100,
        0b_000001000_000010000_000100000,
        0b_001000000_010000000_100000000,
        # yz-diagonal
        0b_001000000_000001000_000000001,
        0b_010000000_000010000_000000010,
        0b_100000000_000100000_000000100,
        0b_000000001_000001000_001000000,
        0b_000000010_000010000_010000000,
        0b_000000100_000100000_100000000,
        # xyz-diagonal
        0b_100000000_000010000_000000001,
        0b_001000000_000010000_000000100,
        0b_000000100_000010000_001000000,
        0b_000000001_000010000_100000000
    ]

    @staticmethod
    def print_board(cells):
        border_line = '+'.join(['---'] * 3)
        print(f'+{border_line}+      +{border_line}+      +{border_line}+')
        for row in range(0, TicTacToe.NUM_ACTIONS, 3):
            board1_row = '|'.join(cells[row:row + 3])
            board2_row = '|'.join(cells[9 + row:9 + row + 3])
            board3_row = '|'.join(cells[18 + row:18 + row + 3])
            print(f'|{board1_row}|      |{board2_row}|      |{board3_row}|')
            print(f'+{border_line}+      +{border_line}+      +{border_line}+')


class TicTacToe3DTree(ContextTree, TicTacToe3D):
    pass
