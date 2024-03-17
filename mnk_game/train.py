import math
from typing import Type
from tqdm import tqdm

import tabular_policies as tp
from policies import MCTSDefaultPolicy
from contexts import Context, ContextTree, ContextPredictor


def fit_q(policy: tp.TabularQPolicy, game: Type[Context], selfplay_count):
    visit_counts = dict()
    progress = tqdm(range(selfplay_count))
    for _ in progress:
        context = game.new()
        rollout = list()
        while not context.done:
            action, _ = policy(context)
            rollout.append((context.board, action))
            context = context(action)
        for board, action in rollout:
            action_visit_counts = visit_counts.setdefault(board, [0] * game.num_actions())
            action_visit_counts[action] += 1
            action_rewards = policy.q_function.get(board, policy.init(game.num_actions()))
            action_rewards[action] += (context.reward - action_rewards[action]) / action_visit_counts[action]
            policy.q_function[board] = action_rewards
        progress.set_postfix(size_q=len(policy.q_function))


def policy_iteration(policy: tp.TabularVPolicy | tp.TabularVUCTPolicy, game: Type[Context],
                     selfplay_count, batch_size, learning_rate):
    batch_count = selfplay_count // batch_size
    history = dict()
    progress = tqdm(range(batch_count))
    for _ in progress:
        batch_dataset = dict()
        for _ in range(batch_size):
            context = game.new()
            rollout = [context.board]
            while not context.done:
                action, _ = policy(context)
                context = context(action)
                rollout.append(context.board)
            for board in rollout:
                batch_dataset.setdefault(board, list()).append(context.reward)
        count = 0
        loss = 0
        for board, rewards in batch_dataset.items():
            state_value = policy.v_function.setdefault(board, policy.init())
            loss += sum((reward - state_value) ** 2 for reward in rewards)
            count += len(rewards)
            policy.v_function[board] += learning_rate * (sum(rewards) / len(rewards) - state_value)
        mean_loss = loss / count
        history.setdefault('loss', list()).append(mean_loss)
        progress.set_postfix(loss=mean_loss)
    return history


def q_policy_iteration(policy: tp.TabularQPolicy, game: Type[Context], selfplay_count, batch_size, learning_rate):
    batch_count = selfplay_count // batch_size
    history = dict()
    progress = tqdm(range(batch_count))
    for _ in progress:
        batch_dataset = dict()
        for _ in range(batch_size):
            context = game.new()
            rollout = list()
            while not context.done:
                action, _ = policy(context)
                rollout.append((context.board, action))
                context = context(action)
            for board, action in rollout:
                batch_dataset.setdefault((board, action), list()).append(context.reward)
        count = 0
        loss = 0
        for (board, action), rewards in batch_dataset.items():
            action_rewards = policy.q_function[board]
            loss += sum((reward - action_rewards[action]) ** 2 for reward in rewards)
            count += len(rewards)
            action_rewards[action] += learning_rate * (sum(rewards) / len(rewards) - action_rewards[action])
            policy.q_function[board] = action_rewards
        mean_loss = loss / count
        history.setdefault('loss', list()).append(mean_loss)
        progress.set_postfix(loss=mean_loss)
    return history


def direct_policy_iteration(policy: MCTSDefaultPolicy, game: Type[ContextTree],
                            selfplay_count, batch_size, learning_rate):
    assert isinstance(policy.default_policy, tp.TabularPiPolicy)
    batch_count = selfplay_count // batch_size
    history = dict()
    progress = tqdm(range(batch_count))
    for _ in progress:
        batch_dataset = dict()
        for _ in range(batch_size):
            context = game.new()
            while not context.done:
                action, _ = policy(context)
                batch_dataset.setdefault(context.board, list()).append(action)
                context = context(action)
        count = 0
        loss = 0
        for board, actions in batch_dataset.items():
            pi, scores = policy.default_policy.pi_function[board]
            loss += -sum(math.log(pi[action]) for action in actions)
            count += len(actions)
            scores = [score - learning_rate * p for score, p in zip(scores, pi)]
            for action in actions:
                scores[action] += learning_rate / len(actions)
            max_score = max(scores)
            weights = [math.exp(score - max_score) for score in scores]
            stat_sum = sum(weights)
            pi = [weight / stat_sum for weight in weights]
            policy.default_policy.pi_function[board] = pi, scores
        mean_loss = loss / count
        history.setdefault('loss', list()).append(mean_loss)
        progress.set_postfix(loss=mean_loss)
    return history


def puct_predictor_iteration(policy: tp.TabularPUCTPolicy, game: Type[ContextPredictor], selfplay_count, batch_size, learning_rate):
    batch_count = selfplay_count // batch_size
    history = dict()
    progress = tqdm(range(batch_count))
    for _ in progress:
        batch_v_dataset = dict()
        batch_pi_dataset = dict()
        for _ in range(batch_size):
            context = game.new()
            rollout = list()
            while not context.done:
                action, _ = policy(context)
                batch_pi_dataset.setdefault(context.board, list()).append(action)
                rollout.append(context.board)
                context = context(action)
            for board in rollout:
                batch_v_dataset.setdefault(board, list()).append(context.reward)
        pi_count = 0
        pi_loss = 0
        for board, actions in batch_pi_dataset.items():
            pi, scores = policy.pi_function[board]
            pi_loss += -sum(math.log(pi[action]) for action in actions)
            pi_count += len(actions)
            scores = [score - learning_rate * p for score, p in zip(scores, pi)]
            for action in actions:
                scores[action] += learning_rate / len(actions)
            max_score = max(scores)
            weights = [math.exp(score - max_score) for score in scores]
            stat_sum = sum(weights)
            pi = [weight / stat_sum for weight in weights]
            policy.pi_function[board] = pi, scores
        mean_pi_loss = pi_loss / pi_count
        pi_size = len(policy.pi_function)
        history.setdefault('pi_loss', list()).append(mean_pi_loss)
        history.setdefault('pi_size', list()).append(pi_size)
        progress.set_postfix(pi_loss=mean_pi_loss, pi_size=pi_size)
    return history


def puct_v_iteration(policy: tp.TabularVTabularPUCTPolicy, game: Type[ContextPredictor], selfplay_count, batch_size, learning_rate):
    batch_count = selfplay_count // batch_size
    history = dict()
    progress = tqdm(range(batch_count))
    for _ in progress:
        batch_v_dataset = dict()
        batch_pi_dataset = dict()
        for _ in range(batch_size):
            context = game.new()
            rollout = list()
            while not context.done:
                action, _ = policy(context)
                batch_pi_dataset.setdefault(context.board, list()).append(action)
                rollout.append(context.board)
                context = context(action)
            for board in rollout:
                batch_v_dataset.setdefault(board, list()).append(context.reward)
        v_count = 0
        v_loss = 0
        for board, rewards in batch_v_dataset.items():
            state_value = policy.v_function[board]
            v_loss += sum((reward - state_value) ** 2 for reward in rewards)
            v_count += len(rewards)
            policy.v_function[board] += learning_rate * (sum(rewards) / len(rewards) - state_value)
        pi_count = 0
        pi_loss = 0
        for board, actions in batch_pi_dataset.items():
            pi, scores = policy.pi_function[board]
            pi_loss += -sum(math.log(pi[action]) for action in actions)
            pi_count += len(actions)
            scores = [score - learning_rate * p for score, p in zip(scores, pi)]
            for action in actions:
                scores[action] += learning_rate / len(actions)
            max_score = max(scores)
            weights = [math.exp(score - max_score) for score in scores]
            stat_sum = sum(weights)
            pi = [weight / stat_sum for weight in weights]
            policy.pi_function[board] = pi, scores
        mean_v_loss = v_loss / v_count
        mean_pi_loss = pi_loss / pi_count
        v_size = len(policy.v_function)
        pi_size = len(policy.pi_function)
        history.setdefault('v_loss', list()).append(mean_v_loss)
        history.setdefault('pi_loss', list()).append(mean_pi_loss)
        history.setdefault('v_size', list()).append(v_size)
        history.setdefault('pi_size', list()).append(pi_size)
        progress.set_postfix(v_loss=mean_v_loss, pi_loss=mean_pi_loss, v_size=v_size, pi_size=pi_size)
    return history
