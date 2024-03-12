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
                return 1, position
        else:
            return 0, None

    @classmethod
    def to_bits(cls, state):
        bits = []
        for _ in range(cls.NUM_ACTIONS):
            bits = bits + [state % 2]
            state = state // 2
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

        done = reward != 0 or sum(board_free) == 0
        actions = list()
        for idx, pos in enumerate(board_free):
            if pos == 1:
                actions.append(idx)

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
        border_line = '+'.join(['---'] * 3)
        print(f'+{border_line}+')
        for row in range(0, 9, 3):
            table_row = '|'.join(cells[row:row + 3])
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

    def print_board(self):
        board_x, board_o = self.board
        _, position_x = self.calculate_reward(board_x)
        _, position_o = self.calculate_reward(board_o)
        cells = self.cells_for_board(board_x, board_o, position_x, position_o, self.actions)
        border_line = '+'.join(['---'] * 3)
        print(f'+{border_line}+      +{border_line}+      +{border_line}+')
        for row in range(0, 9, 3):
            board1_row = '|'.join(cells[row:row + 3])
            board2_row = '|'.join(cells[9 + row:9 + row + 3])
            board3_row = '|'.join(cells[18 + row:18 + row + 3])
            print(f'|{board1_row}|      |{board2_row}|      |{board3_row}|')
            print(f'+{border_line}+      +{border_line}+      +{border_line}+')


class TicTacToe3DTree(ContextTree, TicTacToe3D):
    pass


class UltimateTicTacToe(TicTacToe):

    NUM_SUB_BOARDS = 9

    @classmethod
    def new(cls):
        # (sub_boards_x, sub_boards_o, super_board_x, super_board_o)
        return cls(((0,) * cls.NUM_SUB_BOARDS, (0,) * cls.NUM_SUB_BOARDS, 0, 0))

    @classmethod
    def num_actions(cls):
        return cls.NUM_ACTIONS * cls.NUM_SUB_BOARDS

    def analyze(self):
        sub_boards_x, sub_boards_o, super_board_x, super_board_o = self.board
        x_count = 0
        for sub_board_x in sub_boards_x:
            x_count += sum(self.to_bits(sub_board_x))
            sub_reward_x, _ = self.calculate_reward(sub_board_x)
        o_count = 0
        for sub_board_o in sub_boards_o:
            o_count += sum(self.to_bits(sub_board_o))
            sub_reward_o, _ = self.calculate_reward(sub_board_o)

        reward_x, _ = self.calculate_reward(super_board_x)
        reward_o, _ = self.calculate_reward(super_board_o)
        reward_o = -reward_o

        if x_count == o_count and reward_x == 0:
            move = self.X_MOVE
            reward = reward_o
        elif x_count == o_count + 1 and reward_o == 0:
            move = self.O_MOVE
            reward = reward_x
        else:
            raise ValueError(self.board)

        if self.history:
            last_sub_board = self.history[-1] % self.NUM_SUB_BOARDS
            board_free = self.to_bits(~(sub_boards_x[last_sub_board] | sub_boards_o[last_sub_board]))
            done = reward != 0 or sum(board_free) == 0
            actions = list()
            for idx, pos in enumerate(board_free):
                if pos == 1:
                    actions = actions + [idx + last_sub_board * self.NUM_ACTIONS]
        else:
            actions = list(range(self.NUM_ACTIONS * self.NUM_SUB_BOARDS))
            done = False

        return reward, done, move, actions

    def apply(self, action):
        sub_boards_x, sub_boards_o, super_board_x, super_board_o = self.board
        sub_board_idx = action // self.NUM_SUB_BOARDS
        sub_board_mask = 2 ** sub_board_idx
        if self.move == self.X_MOVE:
            sub_board_x = sub_boards_x[sub_board_idx]
            sub_board_x += 2 ** (action % self.NUM_SUB_BOARDS)
            new_sub_boards_x = list(sub_boards_x)
            new_sub_boards_x[sub_board_idx] = sub_board_x
            if (super_board_x | super_board_o) & sub_board_mask == 0:
                reward_x, _ = self.calculate_reward(sub_board_x)
                super_board_x += reward_x * sub_board_mask
            return tuple(new_sub_boards_x), sub_boards_o, super_board_x, super_board_o
        else:
            sub_board_o = sub_boards_o[sub_board_idx]
            sub_board_o += 2 ** (action % self.NUM_SUB_BOARDS)
            new_sub_boards_o = list(sub_boards_o)
            new_sub_boards_o[sub_board_idx] = sub_board_o
            if (super_board_x | super_board_o) & sub_board_mask == 0:
                reward_o, _ = self.calculate_reward(sub_board_o)
                super_board_o += reward_o * sub_board_mask
            return sub_boards_x, tuple(new_sub_boards_o), super_board_x, super_board_o

    def print_board(self):
        sub_boards_x, sub_boards_o, super_board_x, super_board_o = self.board
        cells: list[str] = list()
        for sub_board_idx, (sub_board_x, sub_board_o) in enumerate(zip(sub_boards_x, sub_boards_o)):
            sub_actions = {action - self.NUM_ACTIONS * sub_board_idx: action for action in self.actions if
                           self.NUM_ACTIONS * sub_board_idx <= action < self.NUM_ACTIONS * (sub_board_idx + 1)}
            sub_board_mask = 2 ** sub_board_idx
            position_x = None
            position_o = None
            if super_board_x & sub_board_mask == sub_board_mask:
                _, position_x = self.calculate_reward(sub_board_x)
            elif super_board_o & sub_board_mask == sub_board_mask:
                _, position_o = self.calculate_reward(sub_board_o)
            cells += self.cells_for_board(sub_board_x, sub_board_o, position_x, position_o, sub_actions)
        print('&&' + '&&'.join(['===&===&==='] * 3) + '&&')
        for sub_board_row in range(3):
            for row in range(3):
                sub_board_rows = list()
                h_separators = list()
                for sub_board_column in range(3):
                    sub_board_idx = sub_board_row * 3 + sub_board_column
                    offset = sub_board_idx * 9+row * 3
                    sub_board_mask = 2 ** sub_board_idx
                    if super_board_x & sub_board_mask == sub_board_mask:
                        v_separator = Fore.RED + '|' + Style.RESET_ALL
                        h_separators.append(Fore.RED + '---+---+---' + Style.RESET_ALL)
                    elif super_board_o & sub_board_mask == sub_board_mask:
                        v_separator = Fore.CYAN + '|' + Style.RESET_ALL
                        h_separators.append(Fore.CYAN + '---+---+---' + Style.RESET_ALL)
                    else:
                        v_separator = '|'
                        h_separators.append('---+---+---')
                    sub_board_rows.append(v_separator.join(cells[offset:offset + 3]))
                print('||' + '||'.join(sub_board_rows) + '||')
                if row < 2:
                    print('&&' + '&&'.join(h_separators) + '&&')
            print('&&' + '&&'.join(['===&===&==='] * 3) + '&&')


class UltimateTicTacToeTree(UltimateTicTacToe, ContextTree):
    pass



