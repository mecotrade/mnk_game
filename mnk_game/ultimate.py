from colorama import Fore, Style

from contexts import ContextTree
from tictactoe import TicTacToe, MNKGame433, MNKGame444


class UltimateTicTacToe(TicTacToe):

    @classmethod
    def new(cls):
        # (sub_boards_x, sub_boards_o, super_board_x, super_board_o)
        return cls(((0,) * cls.NUM_ACTIONS, (0,) * cls.NUM_ACTIONS, 0, 0))

    @classmethod
    def num_actions(cls):
        return cls.NUM_ACTIONS * cls.NUM_ACTIONS

    def calculate_actions(self):
        if self.history:
            sub_boards_x, sub_boards_o, super_board_x, super_board_o = self.board
            target_sub_board_idx = self.history[-1] % self.NUM_ACTIONS
            super_board_mask = 2 ** target_sub_board_idx
            if (super_board_x | super_board_o) & super_board_mask == super_board_mask:
                available_sub_boards = [idx for idx, board in enumerate(self.to_bits(~(super_board_x | super_board_o))) if board == 1]
            else:
                available_sub_boards = [target_sub_board_idx]
            actions = list()
            for sub_board_idx in available_sub_boards:
                sub_board_free = self.to_bits(~(sub_boards_x[sub_board_idx] | sub_boards_o[sub_board_idx]))
                actions += [idx + sub_board_idx * self.NUM_ACTIONS for idx, pos in enumerate(sub_board_free) if pos == 1]
            return actions
        else:
            return list(range(self.num_actions()))

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

        actions = self.calculate_actions() if reward == 0 else list()
        done = len(actions) == 0

        return reward, done, move, actions

    def apply(self, action):
        sub_boards_x, sub_boards_o, super_board_x, super_board_o = self.board
        sub_board_idx = action // self.NUM_ACTIONS
        sub_board_mask = 2 ** sub_board_idx
        if self.move == self.X_MOVE:
            sub_board_x = sub_boards_x[sub_board_idx]
            sub_board_x += 2 ** (action % self.NUM_ACTIONS)
            new_sub_boards_x = list(sub_boards_x)
            new_sub_boards_x[sub_board_idx] = sub_board_x
            if (super_board_x | super_board_o) & sub_board_mask == 0:
                reward_x, _ = self.calculate_reward(sub_board_x)
                super_board_x += reward_x * sub_board_mask
            return tuple(new_sub_boards_x), sub_boards_o, super_board_x, super_board_o
        else:
            sub_board_o = sub_boards_o[sub_board_idx]
            sub_board_o += 2 ** (action % self.NUM_ACTIONS)
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
        sub_board_border = '&'.join(['==='] * self.WIDTH)
        sub_board_separator = '+'.join(['---'] * self.WIDTH)
        print('&&' + '&&'.join([sub_board_border] * self.WIDTH) + '&&')
        for sub_board_row in range(self.HEIGHT):
            for row in range(self.HEIGHT):
                sub_board_rows = list()
                h_separators = list()
                for sub_board_column in range(self.WIDTH):
                    sub_board_idx = sub_board_row * self.WIDTH + sub_board_column
                    offset = sub_board_idx * self.NUM_ACTIONS + row * self.WIDTH
                    sub_board_mask = 2 ** sub_board_idx
                    if super_board_x & sub_board_mask == sub_board_mask:
                        v_separator = Fore.RED + '|' + Style.RESET_ALL
                        h_separators.append(Fore.RED + sub_board_separator + Style.RESET_ALL)
                    elif super_board_o & sub_board_mask == sub_board_mask:
                        v_separator = Fore.CYAN + '|' + Style.RESET_ALL
                        h_separators.append(Fore.CYAN + sub_board_separator + Style.RESET_ALL)
                    else:
                        v_separator = '|'
                        h_separators.append(sub_board_separator)
                    sub_board_rows.append(v_separator.join(cells[offset:offset + self.WIDTH]))
                print('||' + '||'.join(sub_board_rows) + '||')
                if row < self.HEIGHT - 1:
                    print('&&' + '&&'.join(h_separators) + '&&')
            print('&&' + '&&'.join([sub_board_border] * self.WIDTH) + '&&')


class UltimateTicTacToeTree(UltimateTicTacToe, ContextTree):
    pass


class UltimateTicTacToeAlt(UltimateTicTacToe):

    def calculate_actions(self):
        if self.history:
            sub_boards_x, sub_boards_o, _, _ = self.board
            target_sub_board_idx = self.history[-1] % self.NUM_ACTIONS
            sub_board_free = self.to_bits(~(sub_boards_x[target_sub_board_idx] | sub_boards_o[target_sub_board_idx]))
            actions = [idx + target_sub_board_idx * self.NUM_ACTIONS for idx, pos in enumerate(sub_board_free) if pos == 1]
            return actions
        else:
            return list(range(self.num_actions()))


class UltimateTicTacToeAltTree(UltimateTicTacToeAlt, ContextTree):
    pass


class Ultimate433Game(UltimateTicTacToe):
    WIDTH = 4
    HEIGHT = 3
    NUM_ACTIONS = WIDTH * HEIGHT
    WIN_POSITIONS = MNKGame433.WIN_POSITIONS


class Ultimate433GameTree(Ultimate433Game, ContextTree):
    pass


class Ultimate444Game(UltimateTicTacToe):
    WIDTH = 4
    HEIGHT = 4
    NUM_ACTIONS = WIDTH * HEIGHT
    WIN_POSITIONS = MNKGame444.WIN_POSITIONS


class Ultimate444GameTree(Ultimate444Game, ContextTree):
    pass
