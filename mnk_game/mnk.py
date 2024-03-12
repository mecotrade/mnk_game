from colorama import Fore, Style

from contexts import Context, ContextTree


class MNKGame(Context):

    WIDTH = None
    HEIGHT = None
    LINE = None

    X_MOVE = 1
    O_MOVE = -1

    @classmethod
    def new(cls):
        return cls(((0,) * cls.num_actions(), (0,) * cls.num_actions()))

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
                    return 1, (shift, 'horizontal')
        # vertical
        for row in range(cls.HEIGHT - cls.LINE + 1):
            for column in range(cls.WIDTH):
                shift = row * cls.WIDTH + column
                if sum(board[shift:cls.LINE * cls.WIDTH + shift:cls.WIDTH]) == cls.LINE:
                    return 1, (shift, 'vertical')
        # main diagonal
        for row in range(cls.HEIGHT - cls.LINE + 1):
            for column in range(cls.WIDTH - cls.LINE + 1):
                shift = row * cls.WIDTH + column
                if sum(board[cls.LINE - 1 + shift:cls.LINE - 1 + shift + cls.LINE * (cls.WIDTH - 1):cls.WIDTH - 1]) == cls.LINE:
                    return 1, (shift, 'diagonal')
        # anti-diagonal
        for row in range(cls.HEIGHT - cls.LINE + 1):
            for column in range(cls.WIDTH - cls.LINE + 1):
                shift = row * cls.WIDTH + column
                if sum(board[shift:shift + cls.LINE * (cls.WIDTH + 1):cls.WIDTH + 1]) == cls.LINE:
                    return 1, (shift, 'anti-diagonal')
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
            raise ValueError(self.board)

        actions = list()
        for pos, (x, o) in enumerate(zip(board_x, board_o)):
            if x == 0 and o == 0:
                actions.append(pos)

        done = reward != 0 or len(actions) == 0

        return reward, done, move, actions

    def apply(self, action):
        board_x, board_o = self.board
        if self.move == MNKGame.X_MOVE:
            new_board_x = list(board_x)
            new_board_x[action] = 1
            return tuple(new_board_x), board_o
        else:
            new_board_o = list(board_o)
            new_board_o[action] = 1
            return board_x, tuple(new_board_o)

    def add_win_line(self, cells, board, cell):
        _, (shift, line) = self.calculate_reward(board)
        for offset in range(self.LINE):
            if line == 'horizontal':
                cells[shift + offset] = cell
            elif line == 'vertical':
                cells[shift + offset * self.WIDTH] = cell
            elif line == 'diagonal':
                cells[shift + self.LINE - 1 + offset * (self.WIDTH - 1)] = cell
            elif line == 'anti-diagonal':
                cells[shift + offset * (self.WIDTH + 1)] = cell

    def render(self):

        def cell(x, o, pos):
            if x == 1:
                return Fore.RED + ' X ' + Style.RESET_ALL
            elif o == 1:
                return Fore.CYAN + ' O ' + Style.RESET_ALL
            else:
                return Fore.LIGHTBLACK_EX + f'{pos+1:^3}' + Style.RESET_ALL

        board_x, board_o = self.board
        cells = [cell(x, o, pos) for pos, (x, o) in enumerate(zip(board_x, board_o))]

        if self.reward == 1:
            self.add_win_line(cells, board_x, Fore.LIGHTRED_EX + ' # ' + Style.RESET_ALL)
        elif self.reward == -1:
            self.add_win_line(cells, board_o, Fore.LIGHTCYAN_EX + ' @ ' + Style.RESET_ALL)

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


class MNKGame333(MNKGame):
    WIDTH = 3
    HEIGHT = 3
    LINE = 3


class MNKGame333Tree(MNKGame333, ContextTree):
    pass


class MNKGame544(MNKGame):
    WIDTH = 5
    HEIGHT = 4
    LINE = 4


class MNKGame544Tree(ContextTree, MNKGame544):
    pass


class MNKGame554(MNKGame):
    WIDTH = 5
    HEIGHT = 5
    LINE = 4


class MNKGame554Tree(ContextTree, MNKGame554):
    pass
