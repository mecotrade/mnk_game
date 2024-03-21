import torch
import os
import re
import time
import argparse

from typing import Type
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import net_policies
import ultimate
import models
from policies import Policy, RandomPolicy
from contexts import Context
from play import tournament, statistics


def monitor(game: Type[Context], policy: net_policies.NetPolicy, model_folder, opponent: Policy = None):
    already_processed = list()
    event_accumulator = EventAccumulator(model_folder)
    event_accumulator.Reload()
    scalars = event_accumulator.Tags()['scalars']
    if scalars:
        events = event_accumulator.Scalars(scalars[0])
        for event in events:
            already_processed.append(event.step)
            print(f'Step {event.step} is already processed')

    writer = SummaryWriter(model_folder)

    policy.model = torch.load(os.path.join(model_folder, 'model.pt'))
    opponent = opponent or RandomPolicy()
    while True:
        for model_file in os.listdir(model_folder):
            if re.match(r'\d{6}\.pt', model_file):
                step = int(model_file.replace('.pt', ''))
                if step in already_processed:
                    continue
                model_path = os.path.join(model_folder, model_file)
                print()
                print(model_path)
                policy.load(model_path)
                policy.model.eval()
                rewards = tournament(policy, opponent, game, 1000, 5)
                elo, elo_min, elo_max, score, score_min, score_max, score_std, win_rate, loss_rate = statistics(rewards)
                writer.add_scalar('elo/mean', elo, step)
                writer.add_scalar('elo/min', elo_min, step)
                writer.add_scalar('elo/max', elo_max, step)
                writer.add_scalar('score/mean', score, step)
                writer.add_scalar('score/min', score_min, step)
                writer.add_scalar('score/max', score_max, step)
                writer.add_scalar('score/std', score_std, step)
                writer.add_scalar('rate/win', win_rate, step)
                writer.add_scalar('rate/loss', loss_rate, step)
                writer.add_scalar('rate/draw', 1. - win_rate - loss_rate, step)
                already_processed.append(step)
        time.sleep(10)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mf', '--model-folder',
                        help='Folder with model checkpoints')
    args = parser.parse_args()

    game = ultimate.UltimateTicTacToe
    monitor(game, net_policies.GreedyNetVPolicy(models.VModel(game.shape())), args.model_folder)

