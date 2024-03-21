import torch
import numpy as np
import os
import datetime
import random

from torch.utils.tensorboard import SummaryWriter
from typing import Type
from tqdm import tqdm

import policies
import net_policies

from contexts import Context, ContextPredictor


def q_policy_iteration(policy: net_policies.NetQPolicy, opponent: policies.Policy, game: Type[Context], num_games,
                       step_games, step_batches, step_batch_size, buffer_size, learning_rate,
                       save_games=0, upload_score=0.55, run_id=None, num_games_start=0):
    run_id = run_id or datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    writer = SummaryWriter(os.path.join(os.pardir, 'runs', 'q_policy_iteration', run_id))
    model_path = os.path.join(os.pardir, 'models', 'q_policy_iteration', run_id)
    print(model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    optimizer = torch.optim.SGD(policy.model.parameters(), lr=learning_rate)
    history = dict()
    step_count = num_games // step_games
    replay_buffer = list()
    win_rate = 0.5
    elo_opponent = 0
    update_count = 0
    progress = tqdm(range(step_count))
    for step in progress:
        step_rewards = list()
        side = random.choice([game.X_MOVE, game.O_MOVE])
        for _ in range(step_games):
            context = game.new()
            rollout = list()
            while not context.done:
                action, _ = policy(context) if context.move == side else opponent(context)
                rollout.append((context.features(), action))
                context = context(action)

            step_rewards.append(context.reward)
            for features, action in rollout:
                replay_buffer.append((features, action, context.reward))
            side = game.O_MOVE if side == game.X_MOVE else game.X_MOVE

        game_count = step * step_games + num_games_start
        win_rate += 0.01 * ((np.mean(step_rewards) + 1) / 2 - win_rate)
        elo = elo_opponent + 400 * np.log(max(win_rate, 1e-9) / max(1. - win_rate, 1e-9))

        if len(replay_buffer) < buffer_size:
            progress.set_postfix(buffer_size=len(replay_buffer), rewards=step_rewards, games=game_count, win_rate=win_rate, elo=elo)
        else:
            replay_buffer = replay_buffer[-buffer_size:]
            policy.model.cuda()
            losses = list()
            for _ in range(step_batches):
                batch_data = random.choices(replay_buffer, k=step_batch_size)
                batch_features = torch.from_numpy(np.stack([data[0] for data in batch_data])).permute(0, 3, 1, 2).cuda()
                batch_actions = torch.LongTensor([data[1] for data in batch_data])
                batch_rewards = torch.from_numpy(np.array([data[2] for data in batch_data], dtype=np.float32)).cuda()
                q_values = policy.model(batch_features)
                q_values_action = q_values[range(len(batch_actions)), batch_actions]
                loss = torch.nn.functional.mse_loss(q_values_action, batch_rewards)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu().numpy())
            policy.model.cpu()

            loss_value = np.mean(losses)
            history.setdefault('loss', list()).append(loss_value)
            writer.add_scalar('loss/q', loss_value, game_count)
            writer.add_scalar('elo', elo, game_count)
            progress.set_postfix(loss=loss_value, buffer_size=len(replay_buffer), rewards=step_rewards,
                                 games=game_count, win_rate=win_rate, elo=elo, updates=update_count)

            if save_games > 0 and game_count % save_games == 0:
                policy.save(os.path.join(model_path, f'{game_count:06d}.pt'))

            if win_rate > upload_score and isinstance(opponent, type(policy)):
                policy.save(os.path.join(model_path, f'{game_count:06d}.pt'))
                opponent.model.load_state_dict(policy.model.state_dict())
                elo_opponent = elo
                win_rate = 0.5
                update_count += 1

    policy.save(os.path.join(model_path, f'{step_count * step_games + num_games_start:06d}.pt'))
    return history


def policy_iteration(policy: net_policies.NetVPolicy, opponent: policies.Policy, game: Type[Context], num_games,
                     step_games, step_batches, step_batch_size, buffer_size, learning_rate,
                     save_games=0, upload_score=0.55, run_id=None, num_games_start=0):

    run_id = run_id or datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    writer = SummaryWriter(os.path.join(os.pardir, 'runs', 'policy_iteration', run_id))
    model_path = os.path.join(os.pardir, 'models', 'policy_iteration', run_id)
    print(model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(policy.model, os.path.join(model_path, 'model.pt'))

    optimizer = torch.optim.SGD(policy.model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    history = dict()
    step_count = num_games // step_games
    replay_buffer = list()
    win_rate = 0.5
    elo_opponent = 0
    update_count = 0
    progress = tqdm(range(step_count))
    for step in progress:
        policy.model.eval()
        opponent.model.eval()
        step_rewards = list()
        for _ in range(step_games):
            side = random.choice([game.X_MOVE, game.O_MOVE])
            context = game.new()
            rollout = list()
            while not context.done:
                if context.move == side:
                    action, _ = policy(context)
                    rollout.append(context.features())
                else:
                    action, _ = opponent(context)
                context = context(action)

            for features in rollout:
                replay_buffer.append((features, context.reward))
            step_rewards.append(context.reward * side)

        game_count = step * step_games + num_games_start
        win_rate += 0.01 * ((np.mean(step_rewards) + 1) / 2 - win_rate)
        elo = elo_opponent + 400 * np.log(max(win_rate, 1e-9) / max(1. - win_rate, 1e-9))

        replay_buffer = replay_buffer[-buffer_size:]
        policy.model.train()
        policy.model.cuda()
        losses = list()
        for _ in range(step_batches):
            batch_data = random.choices(replay_buffer, k=step_batch_size)
            batch_features = torch.from_numpy(np.stack([data[0] for data in batch_data])).permute(0, 3, 1, 2).cuda()
            batch_rewards = torch.from_numpy(np.array([data[1] for data in batch_data], dtype=np.float32)).cuda()
            values = policy.model(batch_features).reshape([-1])
            loss = torch.nn.functional.mse_loss(values, batch_rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        policy.model.cpu()

        loss_value = np.mean(losses)
        history.setdefault('loss', list()).append(loss_value)
        writer.add_scalar('loss/value', loss_value, game_count)
        writer.add_scalar('elo', elo, game_count)
        writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], game_count)
        progress.set_postfix(loss=loss_value, buffer_size=len(replay_buffer), rewards=step_rewards,
                             games=game_count, win_rate=win_rate, elo=elo, updates=update_count)

        if save_games > 0 and game_count % save_games == 0:
            policy.save(os.path.join(model_path, f'{game_count:06d}.pt'))

        if win_rate > upload_score and isinstance(opponent, type(policy)):
            policy.save(os.path.join(model_path, f'{game_count:06d}.pt'))
            opponent.model.load_state_dict(policy.model.state_dict())
            elo_opponent = elo
            win_rate = 0.5
            update_count += 1

        if game_count > 0 and game_count % 1000 == 0:
            scheduler.step()

    policy.save(os.path.join(model_path, f'{step_count * step_games + num_games_start:06d}.pt'))
    return history


def policy_iteration_selfplay(policy: net_policies.NetVPolicy, game: Type[Context], num_games,
                              step_games, step_batches, step_batch_size, buffer_size, learning_rate,
                              save_games=0, run_id=None, num_games_start=0):

    run_id = run_id or datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    writer = SummaryWriter(os.path.join(os.pardir, 'runs', 'policy_iteration', run_id))
    model_path = os.path.join(os.pardir, 'models', 'policy_iteration', run_id)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(policy.model, os.path.join(model_path, 'model.pt'))

    optimizer = torch.optim.SGD(policy.model.parameters(), lr=learning_rate, weight_decay=1e-3)
    history = dict()
    step_count = num_games // step_games
    replay_buffer = list()
    progress = tqdm(range(step_count))
    for step in progress:
        policy.model.eval()
        step_rewards = list()
        for _ in range(step_games):
            context = game.new()
            rollout = list()
            while not context.done:
                action, _ = policy(context)
                rollout.append(context.features())
                context = context(action)

            for features in rollout:
                replay_buffer.append((features, context.reward))
            step_rewards.append(context.reward)

        game_count = step * step_games + num_games_start

        replay_buffer = replay_buffer[-buffer_size:]
        policy.model.train()
        policy.model.cuda()
        losses = list()
        for _ in range(step_batches):
            batch_data = random.choices(replay_buffer, k=step_batch_size)
            batch_features = torch.from_numpy(np.stack([data[0] for data in batch_data])).permute(0, 3, 1, 2).cuda()
            batch_rewards = torch.from_numpy(np.array([data[1] for data in batch_data], dtype=np.float32)).cuda()
            values = policy.model(batch_features).reshape([-1])
            loss = torch.nn.functional.mse_loss(values, batch_rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        policy.model.cpu()

        loss_value = np.mean(losses)
        history.setdefault('loss', list()).append(loss_value)
        writer.add_scalar('loss/value', loss_value, game_count)
        progress.set_postfix(loss=loss_value, buffer_size=len(replay_buffer), rewards=step_rewards, games=game_count)

        if save_games > 0 and game_count % save_games == 0:
            policy.save(os.path.join(model_path, f'{game_count:06d}.pt'))

    policy.save(os.path.join(model_path, f'{step_count * step_games + num_games_start:06d}.pt'))
    return history


def puct_predictor_iteration(policy: net_policies.NetPUCTDefaultPolicy, game: Type[ContextPredictor], num_games, batch_size, learning_rate,
                             save_games=0, run_id=None, num_games_start=0):
    run_id = run_id or datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    writer = SummaryWriter(os.path.join(os.pardir, 'runs', 'alpha', run_id))
    model_path = os.path.join(os.pardir, 'models', 'alpha', run_id)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    optimizer = torch.optim.Adam(policy.model.parameters(), lr=learning_rate)
    history = dict()
    batch_count = num_games // batch_size
    progress = tqdm(range(batch_count))
    for batch in progress:
        batch_features = list()
        batch_actions = list()
        side = game.X_MOVE if random.random() < 0.5 else game.O_MOVE
        for _ in range(batch_size):
            context = game.new()
            while not context.done:
                action, _ = policy(context)
                batch_features.append(context.features())
                batch_actions.append(action)
                context = context(action)

        logits = policy.model(torch.from_numpy(np.stack(batch_features)).permute(0, 3, 1, 2))
        loss_pi = torch.nn.functional.cross_entropy(logits, torch.nn.functional.one_hot(torch.LongTensor(batch_actions), game.num_actions()).float())
        optimizer.zero_grad()
        loss_pi.backward()
        optimizer.step()

        game_count = batch * batch_size + num_games_start
        loss_pi_value = loss_pi.detach().numpy()
        history.setdefault('loss_pi', list()).append(loss_pi_value)
        writer.add_scalar('loss/pi', loss_pi_value, game_count)
        progress.set_postfix(loss_pi=loss_pi_value)

        if save_games > 0 and game_count % save_games == 0:
            policy.save(os.path.join(model_path, f'{game_count:06d}.pt'))

    policy.save(os.path.join(model_path, f'{batch_count * batch_size + num_games_start:06d}.pt'))
    return history


def alpha_iteration(policy: net_policies.AlphaPolicy, opponent: policies.Policy, game: Type[ContextPredictor], num_games,
                    step_games, step_batches, step_batch_size, buffer_size, learning_rate,
                    save_games=0, upload_score=0.55, run_id=None, num_games_start=0):
    run_id = run_id or datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    writer = SummaryWriter(os.path.join(os.pardir, 'runs', 'alpha', run_id))
    model_path = os.path.join(os.pardir, 'models', 'alpha', run_id)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    optimizer = torch.optim.SGD(policy.model.parameters(), lr=learning_rate, weight_decay=1e-3)
    history = dict()
    step_count = num_games // step_games
    replay_buffer = list()
    win_rate = 0.5
    elo_opponent = 0
    update_count = 0
    progress = tqdm(range(step_count))
    for step in progress:
        policy.model.eval()
        opponent.model.eval()
        step_rewards = list()
        for _ in range(step_games):
            side = random.choice([game.X_MOVE, game.O_MOVE])
            context = game.new()
            rollout = list()
            while not context.done:
                if context.move == side:
                    action, _ = policy(context)
                    rollout.append((context.features(), action))
                else:
                    action, _ = opponent(context)
                context = context(action)

            step_rewards.append(context.reward)
            for features, action in rollout:
                replay_buffer.append((features, action, context.reward))

        game_count = step * step_games + num_games_start
        win_rate += 0.01 * ((np.mean(step_rewards) + 1) / 2 - win_rate)
        elo = elo_opponent + 400 * np.log(max(win_rate, 1e-6) / max(1. - win_rate, 1e-6))

        replay_buffer = replay_buffer[-buffer_size:]
        model = policy.model.cuda()
        losses_v = list()
        losses_pi = list()
        for _ in range(step_batches):
            batch_data = random.choices(replay_buffer, k=step_batch_size)
            batch_features = torch.from_numpy(np.stack([data[0] for data in batch_data])).permute(0, 3, 1, 2).cuda()
            batch_actions = torch.nn.functional.one_hot(torch.LongTensor([data[1] for data in batch_data]), game.num_actions()).float().cuda()
            batch_rewards = torch.from_numpy(np.array([data[2] for data in batch_data], dtype=np.float32)).cuda()
            logits, values = model(batch_features)
            loss_v = torch.nn.functional.mse_loss(torch.reshape(values, [-1]), batch_rewards)
            loss_pi = torch.nn.functional.cross_entropy(logits, batch_actions)
            loss = loss_v + loss_pi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_v.append(loss_v.detach().cpu().numpy())
            losses_pi.append(loss_pi.detach().cpu().numpy())
        policy.model = model.cpu()

        loss_v_value = np.mean(losses_v)
        loss_pi_value = np.mean(losses_pi)
        history.setdefault('loss_v', list()).append(loss_v_value)
        history.setdefault('loss_pi', list()).append(loss_pi_value)
        writer.add_scalar('loss/value', loss_v_value, game_count)
        writer.add_scalar('loss/pi', loss_pi_value, game_count)
        writer.add_scalar('elo', elo, game_count)
        progress.set_postfix(loss_pi=loss_pi_value, loss_v=loss_v_value, buffer_size=len(replay_buffer),
                             rewards=step_rewards, games=game_count, win_rate=win_rate, elo=elo, updates=update_count)

        if save_games > 0 and game_count % save_games == 0:
            policy.save(os.path.join(model_path, f'{game_count:06d}.pt'))

        if win_rate > upload_score and isinstance(opponent, type(policy)):
            opponent.model.load_state_dict(policy.model.state_dict())
            elo_opponent = elo
            win_rate = 0.5
            update_count += 1
            replay_buffer.clear()

    policy.save(os.path.join(model_path, f'{step_count * step_games + num_games_start:06d}.pt'))
    return history
