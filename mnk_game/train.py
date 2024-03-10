from typing import Type
from tqdm import tqdm

from policies import TabularQPolicy
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
            action_numbers = numbers.setdefault(board, [0] * context.num_actions())
            action_numbers[action] += 1
            action_rewards = policy.q_function.setdefault(board, [0.] * context.num_actions())
            action_rewards[action] += (context.reward - action_rewards[action]) / action_numbers[action]
            policy.q_function[board] = action_rewards
