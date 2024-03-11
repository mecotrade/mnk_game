import math
from typing import Type
from tqdm import tqdm

from policies import TabularQPolicy, TabularVPolicy, TabularPiPolicy, MCTSPolicy
from contexts import Context


def fit_q(policy: TabularQPolicy, game: Type[Context], selfplay_count):
    visit_counts = dict()
    for _ in tqdm(range(selfplay_count)):
        context = game.new()
        rollout = list()
        while not context.done:
            action, _ = policy(context)
            rollout.append((context.board, action))
            context = context(action)
        for board, action in rollout:
            action_visit_counts = visit_counts.setdefault(board, [0] * game.num_actions())
            action_visit_counts[action] += 1
            action_rewards = policy.q_function.setdefault(board, [0.] * game.num_actions())
            action_rewards[action] += (context.reward - action_rewards[action]) / action_visit_counts[action]
            policy.q_function[board] = action_rewards


def policy_iteration(policy: TabularVPolicy, game: Type[Context], selfplay_count, batch_size, learning_rate):
    batch_count = selfplay_count // batch_size
    history = list()
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
            state_value = policy.v_function.setdefault(board, 0.)
            policy.v_function[board] += learning_rate * (sum(rewards) / len(rewards) - state_value)
            loss += sum((reward - state_value) ** 2 for reward in rewards)
            count += len(rewards)
        mean_loss = loss / count
        history.append(mean_loss)
        progress.set_postfix(loss=mean_loss)
    return history


def q_policy_iteration(policy: TabularQPolicy, game: Type[Context], selfplay_count, batch_size, learning_rate):
    batch_count = selfplay_count // batch_size
    history = list()
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
            action_rewards = policy.q_function.setdefault(board, [0.] * game.num_actions())
            action_rewards[action] += learning_rate * (sum(rewards) / len(rewards) - action_rewards[action])
            policy.q_function[board] = action_rewards
            loss += sum((reward - action_rewards[action]) ** 2 for reward in rewards)
            count += len(rewards)
        mean_loss = loss / count
        history.append(mean_loss)
        progress.set_postfix(loss=mean_loss)
    return history


def direct_policy_iteration(policy: MCTSPolicy, game: Type[Context], selfplay_count, batch_size, learning_rate):
    assert isinstance(policy.default_policy, TabularPiPolicy)
    batch_count = selfplay_count // batch_size
    history = list()
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
            pi, scores = policy.default_policy.pi_function.setdefault(board, ([1/game.num_actions()]*game.num_actions(),
                                                                              [0]*game.num_actions()))
            scores = [score - learning_rate * p for score, p in zip(scores, pi)]
            for action in actions:
                scores[action] += learning_rate / len(actions)
            max_score = max(scores)
            weights = [math.exp(score - max_score) for score in scores]
            stat_sum = sum(weights)
            pi = [weight / stat_sum for weight in weights]
            policy.default_policy.pi_function[board] = pi, scores
            loss += -sum(math.log(pi[action]) for action in actions)
            count += len(actions)
        mean_loss = loss / count
        history.append(mean_loss)
        progress.set_postfix(loss=mean_loss)
    return history



