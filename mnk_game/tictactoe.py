from colorama import Fore, Style

from contexts import Context, ContextTree


class TicTacToe(Context):

    X_MOVE = 1
    O_MOVE = -1

    WIDTH = 3
    HEIGHT = 3
    NUM_ACTIONS = WIDTH * HEIGHT

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
                return 1, position
        else:
            return 0, None

    @classmethod
    def to_bits(cls, board):
        bits = list()
        for _ in range(cls.NUM_ACTIONS):
            bits.append(board % 2)
            board //= 2
        return bits

    def analyze(self):
        board_x, board_o = self.board
        reward_x, _ = self.calculate_reward(board_x)
        reward_o, _ = self.calculate_reward(board_o)
        reward_o = -reward_o

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
            raise ValueError(self.board)

        board_free = self.to_bits(~(board_x | board_o))

        actions = [idx for idx, pos in enumerate(board_free) if pos == 1] if reward == 0 else list()
        done = len(actions) == 0

        return reward, done, move, actions

    def apply(self, action):
        board_x, board_o = self.board
        if self.move == TicTacToe.X_MOVE:
            board_x += 2 ** action
        else:
            board_o += 2 ** action
        return board_x, board_o

    @classmethod
    def cells_for_board(cls, board_x, board_o, position_x, position_o, actions):
        bits_x = cls.to_bits(board_x)
        bits_o = cls.to_bits(board_o)
        cells = list()
        for pos, (x, o) in enumerate(zip(bits_x, bits_o)):
            if x == 1:
                cells.append(Fore.RED + ' X ' + Style.RESET_ALL)
            elif o == 1:
                cells.append(Fore.CYAN + ' O ' + Style.RESET_ALL)
            elif pos in actions:
                action = actions[pos] + 1 if isinstance(actions, dict) else pos + 1
                cells.append(Fore.LIGHTBLACK_EX + f'{action:^3}' + Style.RESET_ALL)
            else:
                cells.append('   ')
        if position_x is not None:
            bits_x = cls.to_bits(position_x)
            for idx, pos in enumerate(bits_x):
                if pos == 1:
                    cells[idx] = Fore.RED + ' # ' + Style.RESET_ALL
        elif position_o is not None:
            bits_o = cls.to_bits(position_o)
            for idx, pos in enumerate(bits_o):
                if pos == 1:
                    cells[idx] = Fore.CYAN + ' @ ' + Style.RESET_ALL
        return cells

    def print_board(self):
        board_x, board_o = self.board
        _, position_x = self.calculate_reward(board_x)
        _, position_o = self.calculate_reward(board_o)
        cells = self.cells_for_board(board_x, board_o, position_x, position_o, self.actions)
        border_line = '+'.join(['---'] * self.WIDTH)
        print(f'+{border_line}+')
        for row in range(self.HEIGHT):
            table_row = '|'.join(cells[row * self.WIDTH:row * self.WIDTH + self.WIDTH])
            print(f'|{table_row}|')
            print(f'+{border_line}+')

    def render(self):
        self.print_board()
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


class MNKGame433(TicTacToe):
    WIDTH = 4
    HEIGHT = 3
    NUM_ACTIONS = WIDTH * HEIGHT
    WIN_POSITIONS = [
        0b_0000_0000_0111,
        0b_0000_0000_1110,
        0b_0000_0111_0000,
        0b_0000_1110_0000,
        0b_0111_0000_0000,
        0b_1110_0000_0000,
        0b_0001_0001_0001,
        0b_0010_0010_0010,
        0b_0100_0100_0100,
        0b_1000_1000_1000,
        0b_0100_0010_0001,
        0b_1000_0100_0010,
        0b_0010_0100_1000,
        0b_0001_0010_0100
    ]


class MNKGame433Tree(MNKGame433, ContextTree):
    pass


class MNKGame444(TicTacToe):
    WIDTH = 4
    HEIGHT = 4
    NUM_ACTIONS = WIDTH * HEIGHT
    WIN_POSITIONS = [
        0b_0000_0000_0000_1111,
        0b_0000_0000_1111_0000,
        0b_0000_1111_0000_0000,
        0b_1111_0000_0000_0000,
        0b_0001_0001_0001_0001,
        0b_0010_0010_0010_0010,
        0b_0100_0100_0100_0100,
        0b_1000_1000_1000_1000,
        0b_1000_0100_0010_0001,
        0b_0001_0010_0100_1000
    ]


class MNKGame444Tree(MNKGame444, ContextTree):
    pass
