import numpy as np
import random
import os


from typing import Type
from tqdm import tqdm

import ultimate
import net_policies
import models

from contexts import Context, ContextTree
from policies import Policy


def print_info(info, actions):
    for key, value in info.items():
        if isinstance(value, dict):
            print(f'{key}:')
            for action, score in sorted(value.items(), key=lambda entry: entry[1], reverse=True):
                if action in actions:
                    print(f'\t{action + 1}: {score}')
        else:
            print(f'{key}: {value}')


def play(policy: Policy, side, game: Type[Context], verbose=False):
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
                    print_info(info, context.actions)
                print(f'Made action {action+1}')
            else:
                action = None
                while action not in context.actions:
                    if isinstance(context, ContextTree):
                        human_input = input(f'Choose your move, "?" for hint, or "q" for quit: ')
                        if human_input == '?':
                            values = dict()
                            visits = dict()
                            for hint_action, child in enumerate(context.children):
                                if child is not None:
                                    values[hint_action] = child.value
                                    visits[hint_action] = child.visits
                            print_info({'values': values, 'visits': visits}, context.actions)
                            continue
                    else:
                        human_input = input(f'Choose your move, or "q" for quit: ')
                    if human_input == 'q':
                        print('Buy-buy!')
                        return
                    elif human_input.isdigit():
                        action = int(human_input) - 1
                        if action not in context.actions:
                            print('Wrong move, try again')
                    else:
                        print('Wrong input, try again')
            context = context(action)
            context.render()
        if context.reward == 1:
            win_x += 1
        elif context.reward == -1:
            win_o += 1
        else:
            win_x += 0.5
            win_o += 0.5


def play_against(policy_x: Policy, policy_o: Policy, game: Type[Context], num_games):
    rewards = list()
    for _ in range(num_games):
        context = game.new()
        while not context.done:
            action, _ = policy_x(context)
            context = context(action)
            if not context.done:
                action, _ = policy_o(context)
                context = context(action)
        rewards.append(context.reward)
    return rewards


def statistics(rewards):
    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards, ddof=1)
    score = (1 + reward_mean) / 2
    score_std = (reward_std / 2) / np.sqrt(len(rewards))
    score_min = score - 2 * score_std
    score_max = score + 2 * score_std
    elo = 400 * np.log(max(score, 1e-9) / max(1. - score, 1e-9))
    elo_min = 400 * np.log(max(score_min, 1e-9) / max(1. - score_min, 1e-9))
    elo_max = 400 * np.log(max(score_max, 1e-9) / max(1. - score_max, 1e-9))
    win_rate = np.mean(np.array(rewards) == 1)
    loss_rate = np.mean(np.array(rewards) == -1)

    return elo, elo_min, elo_max, score, score_min, score_max, score_std, win_rate, loss_rate


def tournament(policy: Policy, opponent: Policy, game: Type[Context], num_games, tournament_games, elo_op=0):
    tournament_count = num_games // tournament_games
    progress = tqdm(range(tournament_count))
    rewards = list()
    for _ in progress:
        for _ in range(tournament_games):
            context = game.new()
            side = random.choice([game.X_MOVE, game.O_MOVE])
            while not context.done:
                action, _ = policy(context) if context.move == side else opponent(context)
                context = context(action)
            rewards.append(context.reward * side)
        elo, elo_min, elo_max, score, score_min, score_max, score_std, win_rate, loss_rate = statistics(rewards)
        progress.set_postfix(elo=elo, elo_min=elo_min, elo_max=elo_max,
                             score=score, wins=win_rate, losses=loss_rate,
                             std=score_std)
    return rewards
