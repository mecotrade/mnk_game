from typing import Type

from contexts import Context


def print_info(info):
    for key, value in info.items():
        if isinstance(value, dict):
            print(f'{key}:')
            for k, v in sorted(value.items(), key=lambda entry: entry[1], reverse=True):
                print(f'\t{k+1}: {v}')
        else:
            print(f'{key}: {value}')


def play(policy, side, game: Type[Context], verbose=False):
    win_x = 0
    win_o = 0
    while True:
        title = f'Crosses: {win_x}, Noughts: {win_o}'
        print('=' * len(title))
        print(title)
        print('=' * len(title))
        context = game.new()
        if context.move != side:
            context.render()
        while not context.done:
            if context.move == side:
                action, info = policy(context)
                if verbose:
                    print_info(info)
                print(f'Made action {action+1}')
            else:
                action = None
                while action not in context.actions:
                    human_input = input(f'Choose your move, or "q" for quit: ')
                    if human_input == 'q':
                        print('Buy-buy!')
                        return
                    else:
                        action = int(human_input) - 1
                        if action not in context.actions:
                            print('Wrong move, try again')
            context = context(action)
            context.render()
        if context.reward == 1:
            win_x += 1
        elif context.reward == -1:
            win_o += 1
        else:
            win_x += 0.5
            win_o += 0.5
