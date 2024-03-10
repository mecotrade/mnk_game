from typing import Type
from tqdm import tqdm

from policies import TabularQPolicy, TabularVPolicy
from contexts import Context


def fit_q(policy: TabularQPolicy, game: Type[Context], num_rollouts):
    numbers = dict()
    for _ in tqdm(range(num_rollouts)):
        context = game.new()
        rollout = list()
        while not context.done:
            action, _ = policy(context)
            rollout.append((context.board, action))
            context = context(action)
        for board, action in rollout:
            action_numbers = numbers.setdefault(board, [0] * game.num_actions())
            action_numbers[action] += 1
            action_rewards = policy.q_function.setdefault(board, [0.] * game.num_actions())
            action_rewards[action] += (context.reward - action_rewards[action]) / action_numbers[action]
            policy.q_function[board] = action_rewards


def policy_iteration(policy: TabularVPolicy, game: Type[Context], num_rollouts, batch_size, learning_rate):
    num_batches = num_rollouts // batch_size
    for _ in tqdm(range(num_batches)):
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


def q_policy_iteration(policy: TabularQPolicy, game: Type[Context], num_rollouts, batch_size, learning_rate):
    num_batches = num_rollouts // batch_size
    for _ in tqdm(range(num_batches)):
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


