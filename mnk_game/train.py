from typing import Type
from tqdm import tqdm

from policies import TabularQPolicy, TabularVPolicy
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
    for _ in tqdm(range(batch_count)):
        history = dict()
        for _ in range(batch_size):
            context = game.new()
            rollout = [context.board]
            while not context.done:
                action, _ = policy(context)
                context = context(action)
                rollout.append(context.board)
            for board in rollout:
                history.setdefault(board, list()).append(context.reward)
        for board, rewards in history.items():
            state_value = policy.v_function.setdefault(board, 0.)
            policy.v_function[board] += learning_rate * (sum(rewards) / len(rewards) - state_value)


def q_policy_iteration(policy: TabularQPolicy, game: Type[Context], selfplay_count, batch_size, learning_rate):
    batch_count = selfplay_count // batch_size
    for _ in tqdm(range(batch_count)):
        history = dict()
        for _ in range(batch_size):
            context = game.new()
            rollout = list()
            while not context.done:
                action, _ = policy(context)
                rollout.append((context.board, action))
                context = context(action)
            for board, action in rollout:
                history.setdefault((board, action), list()).append(context.reward)
        for (board, action), rewards in history.items():
            action_rewards = policy.q_function.setdefault(board, [0.] * game.num_actions())
            action_rewards[action] += learning_rate * (sum(rewards) / len(rewards) - action_rewards[action])
            policy.q_function[board] = action_rewards


