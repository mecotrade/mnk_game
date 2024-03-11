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
        x_count = sum(bits_x)
        o_count = sum(bits_o)
        if x_count == o_count and reward_x == 0:
            move = TicTacToe.X_MOVE
            reward = reward_o
        elif x_count == o_count + 1 and reward_o == 0:
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


class TicTacToeTree(ContextTree, TicTacToe):
    pass


class MNKGame(Context):

    WIDTH = None
    HEIGHT = None
    LINE = None

    X_MOVE = 1
    O_MOVE = -1

    @classmethod
    def new(cls):
        return cls(([0] * cls.num_actions(), [0] * cls.num_actions()))

    @classmethod
    def num_actions(cls):
        return cls.WIDTH * cls.HEIGHT

    @classmethod
    def calculate_reward(cls, board):
        # horizontal
        for row in range(cls.HEIGHT):
            for column in range(cls.WIDTH - cls.LINE + 1):
                shift = row * cls.WIDTH + column
                if sum(board[shift:shift + cls.LINE]) == cls.LINE:
                    return 1, (row, column, 'horizontal')
        # vertical
        for row in range(cls.HEIGHT - cls.LINE + 1):
            for column in range(cls.WIDTH):
                shift = row * cls.WIDTH + column
                if sum(board[shift:cls.LINE * cls.WIDTH + shift:cls.WIDTH]) == cls.LINE:
                    return 1, (row, column, 'vertical')
        # main diagonal
        for row in range(cls.HEIGHT - cls.LINE + 1):
            for column in range(cls.WIDTH - cls.LINE + 1):
                shift = row * cls.WIDTH + column
                if sum(board[cls.LINE - 1 + shift:cls.LINE - 1 + shift + cls.LINE * (cls.WIDTH - 1):cls.WIDTH - 1]) == cls.LINE:
                    return 1, (row, column, 'main diagonal')
        # anti-diagonal
        for row in range(cls.HEIGHT - cls.LINE + 1):
            for column in range(cls.WIDTH - cls.LINE + 1):
                shift = row * cls.WIDTH + column
                if sum(board[shift:shift + cls.LINE * (cls.WIDTH + 1):cls.WIDTH + 1]) == cls.LINE:
                    return 1, (row, column, 'anti-diagonal')
        return 0, None

    def analyze(self):
        board_x, board_o = self.board
        x_count = sum(board_x)
        o_count = sum(board_o)

        reward_x, _ = self.calculate_reward(board_x)
        reward_o, _ = self.calculate_reward(board_o)
        reward_o = -reward_o

        if x_count == o_count and reward_x == 0:
            move = MNKGame.X_MOVE
            reward = reward_o
        elif x_count == o_count + 1 and reward_o == 0:
            move = MNKGame.O_MOVE
            reward = reward_x
        else:
            move = None
            reward = 0

        actions = list()
        for pos, (x, o) in enumerate(zip(board_x, board_o)):
            if x == 0 and o == 0:
                actions.append(pos)

        done = reward != 0 or len(actions) == 0

        return reward, done, move, actions

    def apply(self, action):
        board_x, board_o = self.board
        board_x = board_x.copy()
        board_o = board_o.copy()
        if self.move == MNKGame.X_MOVE:
            board_x[action] = 1
        else:
            board_o[action] = 1
        return board_x, board_o

    def render(self):

        def cell(x, o, pos):
            if x == 1:
                return Fore.RED + ' X ' + Style.RESET_ALL
            elif o == 1:
                return Fore.BLUE + ' O ' + Style.RESET_ALL
            else:
                return Fore.YELLOW + f'{pos+1:^3}' + Style.RESET_ALL

        board_x, board_o = self.board
        cells = [cell(x, o, pos) for pos, (x, o) in enumerate(zip(board_x, board_o))]

        def add_win_line(cells, board):
            _, (row, column, line) = self.calculate_reward(board)
            if self.reward == 1:
                cell = Fore.LIGHTMAGENTA_EX + f' X ' + Style.RESET_ALL
            else:
                cell = Fore.LIGHTCYAN_EX + f' O ' + Style.RESET_ALL
            for offset in range(self.LINE):
                if line == 'horizontal':
                    cells[row * self.WIDTH + column + offset] = cell
                elif line == 'vertical':
                    cells[(row + offset) * self.WIDTH + column] = cell
                elif line == 'main diagonal':
                    cells[row * self.WIDTH + column + self.LINE - 1 + offset * (self.WIDTH - 1)] = cell
                elif line == 'anti-diagonal':
                    cells[row * self.WIDTH + column + offset * (self.WIDTH + 1)] = cell

        if self.reward == 1:
            add_win_line(cells, board_x)
        elif self.reward == -1:
            add_win_line(cells, board_o)

        border_line = '+'.join(['---'] * self.WIDTH)
        print(f'+{border_line}+')
        for row in range(0, len(cells), self.WIDTH):
            table_row = '|'.join(cells[row:row + self.WIDTH])
            print(f'|{table_row}|')
            print(f'+{border_line}+')

        if not self.done:
            if self.move == MNKGame.X_MOVE:
                print('Crosses move')
            elif self.move == MNKGame.O_MOVE:
                print('Noughts move')
        else:
            if self.reward == 1:
                print('Crosses win')
            elif self.reward == -1:
                print('Noughts win')
            else:
                print('Draw')


class MNKGame544(MNKGame):
    WIDTH = 5
    HEIGHT = 4
    LINE = 4


class MNKGame554(MNKGame):
    WIDTH = 5
    HEIGHT = 5
    LINE = 4


class MNKGame544Tree(ContextTree, MNKGame544):
    pass


class MNKGame554Tree(ContextTree, MNKGame554):
    pass

