import random
import math
from typing import Type

from contexts import Context, ContextTree


class Policy:

    def __call__(self, context: Context) -> (int, dict):
        raise NotImplementedError


class RandomPolicy(Policy):

    @staticmethod
    def apply(context: Context):
        return random.choice(context.actions), {'policy': 'random'}

    def __call__(self, context: Context):
        return RandomPolicy.apply(context)


class ScorePolicy(Policy):

    def scores(self, context: Context) -> list:
        raise NotImplementedError


class TabularQPolicy(ScorePolicy):

    def __init__(self, q_function=None):
        self.q_function = q_function or dict()

    def scores(self, context: Context):
        q_values = self.q_function.get(context.board)
        if q_values is not None:
            q_values = [q_values[action] * context.move for action in context.actions]
        else:
            q_values = [0.] * len(context.actions)
        return q_values


class TabularVPolicy(ScorePolicy):

    def __init__(self, v_function=None):
        self.v_function = v_function or dict()

    def scores(self, context: Context):
        values = list()
        for action in context.actions:
            virtual_board = context.apply(action)
            value = self.v_function.get(virtual_board, 0.)
            values.append(value * context.move)
        return values


class TabularPiPolicy(ScorePolicy):

    def __init__(self, pi_function=None):
        self.pi_function = pi_function or dict()

    def scores(self, context: Context):
        pi_scores = self.pi_function.get(context.board)
        if pi_scores is not None:
            _, scores = pi_scores
            return [scores[action] for action in context.actions]
        else:
            return [0.] * len(context.actions)

    @staticmethod
    def uniform(context: Context | Type[Context]):
        n = context.num_actions()
        return [1/n] * n, [0.] * n


class GreedyPolicy(ScorePolicy):

    def __call__(self, context: Context):
        scores = self.scores(context)
        max_score = None
        best_actions = None
        action_scores = dict()
        for action, score in zip(context.actions, scores):
            action_scores[action] = score
            if max_score is None or max_score < score:
                max_score = score
                best_actions = [action]
            elif max_score == score:
                best_actions.append(action)
        policy_action = random.choice(best_actions)
        return policy_action, {'policy': 'greedy', 'scores': action_scores}


class EpsilonGreedyPolicy(GreedyPolicy):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, context: Context):
        if random.random() < self.epsilon:
            return RandomPolicy.apply(context)
        else:
            return super().__call__(context)


class BoltzmannPolicy(ScorePolicy):
    def __init__(self, temperature=1.):
        self.temperature = temperature

    def __call__(self, context: Context):
        scores = self.scores(context)
        max_score = max(scores)
        weights = [math.exp((score - max_score) / self.temperature) for score in scores]
        stat_sum = sum(weights)
        action_proba = [weight / stat_sum for weight in weights]
        policy_action = random.choices(context.actions, action_proba)[0]
        return policy_action, {
            'policy': 'boltzmann',
            'scores': {action: score for action, score in zip(context.actions, scores)},
            'probability': {action: proba for action, proba in zip(context.actions, action_proba)}
        }


class GreedyTabularQPolicy(GreedyPolicy, TabularQPolicy):

    def __init__(self, v_function=None):
        TabularQPolicy.__init__(self, v_function)


class GreedyTabularVPolicy(GreedyPolicy, TabularVPolicy):

    def __init__(self, v_function=None):
        TabularVPolicy.__init__(self, v_function)


class GreedyTabularPiPolicy(GreedyPolicy, TabularPiPolicy):

    def __init__(self, pi_function=None):
        TabularPiPolicy.__init__(self, pi_function)


class EpsilonGreedyTabularQPolicy(EpsilonGreedyPolicy, TabularQPolicy):

    def __init__(self, epsilon, q_function=None):
        EpsilonGreedyPolicy.__init__(self, epsilon)
        TabularQPolicy.__init__(self, q_function)


class BoltzmannTabularVPolicy(BoltzmannPolicy, TabularVPolicy):

    def __init__(self, temperature=1., v_function=None):
        BoltzmannPolicy.__init__(self, temperature)
        TabularVPolicy.__init__(self, v_function)


class BoltzmannTabularQPolicy(BoltzmannPolicy, TabularQPolicy):

    def __init__(self, temperature=1., q_function=None):
        BoltzmannPolicy.__init__(self, temperature)
        TabularQPolicy.__init__(self, q_function)


class BoltzmannTabularPiPolicy(BoltzmannPolicy, TabularPiPolicy):

    def __init__(self, temperature=1., pi_function=None):
        BoltzmannPolicy.__init__(self, temperature)
        TabularPiPolicy.__init__(self, pi_function)


class TreePolicy:

    def expand(self, context: ContextTree):
        raise NotImplementedError


class MCTSPolicy(Policy, TreePolicy):

    def __init__(self, rollout_count, c=1, temperature=1, use_visits=False):
        self.rollout_count = rollout_count
        self.c = c
        self.temperature = temperature
        self.use_visits = use_visits

    def select(self, context: ContextTree):
        while True:
            context.visits += 1
            if context.done:
                return context.history, context.reward
            unexplored = [action for action in context.actions if context.children[action] is None]
            if len(unexplored) > 0:
                action = random.choice(unexplored)
                child = context(action)
                child.visits += 1
                return child.history, self.expand(child)
            max_bound = None
            best_actions = None
            for action in context.actions:
                child = context.children[action]
                child_bound = (context.move / child.move) * child.value + self.c * math.sqrt(math.log(context.visits) / child.visits)
                if max_bound is None or child_bound > max_bound:
                    max_bound = child_bound
                    best_actions = [action]
                elif child_bound == max_bound:
                    best_actions.append(action)
            selected_action = random.choice(best_actions)
            context = context(selected_action)

    @staticmethod
    def backward(context, history, reward):
        for action in history:
            child = context(action)
            child.value += (reward * child.move - child.value) / child.visits
            context = child

    def __call__(self, context: ContextTree):
        for _ in range(self.rollout_count):
            history, reward = self.select(context)
            MCTSPolicy.backward(context, history[len(context.history):], reward)
        actions = list()
        action_values = list()
        action_visits = list()
        for action in context.actions:
            child: ContextTree = context.children[action]
            if child is not None:
                actions.append(action)
                action_values.append(context.move / child.move * child.value)
                action_visits.append(child.visits)
        if self.use_visits:
            max_visits = max(action_visits)
            weights = [(visits / max_visits) ** (1 / self.temperature) for visits in action_visits]
        else:
            max_value = max(action_values)
            weights = [math.exp((value - max_value) / self.temperature) for value in action_values]
        stat_sum = sum(weights)
        action_proba = [weight / stat_sum for weight in weights]
        policy_action = random.choices(actions, action_proba)[0]
        return policy_action, {
            'policy': 'mcts',
            'values': {action: value for action, value in zip(actions, action_values)},
            'visits': {action: visits for action, visits in zip(actions, action_visits)},
            'probability': {action: proba for action, proba in zip(actions, action_proba)}
        }


class PUCTPolicy(MCTSPolicy):

    def __init__(self, rollout_count, c=1, temperature=1, use_visits=False, pi_function=None):
        super().__init__(rollout_count, c, temperature, use_visits)
        self.pi_function = pi_function or dict()

    def select(self, context: ContextTree):
        while True:
            context.visits += 1
            if context.done:
                return context.history, context.reward
            max_bound = None
            best_actions = None
            pi, _ = self.pi_function.get(context.board, TabularPiPolicy.uniform(context))
            for action in context.actions:
                child = context(action)
                child_bound = (context.move / child.move) * child.value + self.c * pi[action] * math.sqrt(context.visits) / (child.visits + 1)
                if max_bound is None or child_bound > max_bound:
                    max_bound = child_bound
                    best_actions = [action]
                elif child_bound == max_bound:
                    best_actions.append(action)
            selected_action = random.choice(best_actions)
            context = context(selected_action)
            if context.visits == 0:
                context.visits += 1
                return context.history, self.expand(context)

    def __call__(self, context):
        action, info = super().__call__(context)
        pi, scores = self.pi_function.get(context.board, TabularPiPolicy.uniform(context))
        info['policy'] = 'puct'
        info['pi'] = {action: pi[action] for action in context.actions}
        info['scores'] = {action: scores[action] for action in context.actions}
        return action, info


class DefaultTreePolicy(TreePolicy):

    def __init__(self, default_policy=None):
        self.default_policy = default_policy or RandomPolicy()

    def expand(self, context: ContextTree):
        while not context.done:
            action, _ = self.default_policy(context)
            context = context.of(action)
        return context.reward


class MCTSDefaultPolicy(MCTSPolicy, DefaultTreePolicy):

    def __init__(self, rollout_count, c=1, temperature=1, use_visits=False, default_policy=None):
        MCTSPolicy.__init__(self, rollout_count, c, temperature, use_visits)
        DefaultTreePolicy.__init__(self, default_policy)


class PUCTDefaultPolicy(PUCTPolicy, DefaultTreePolicy):

    def __init__(self, rollout_count, c=1, temperature=1, use_visits=False, pi_function=None, default_policy=None):
        PUCTPolicy.__init__(self, rollout_count, c, temperature, use_visits, pi_function)
        DefaultTreePolicy.__init__(self, default_policy)


class TabularTreePolicy(TreePolicy):

    def __init__(self, v_function=None):
        self.v_function = v_function or dict()

    def expand(self, context):
        return context.reward if context.done else self.v_function.get(context.board, 0.)


class TabularUCTPolicy(MCTSPolicy, TabularTreePolicy):

    def __init__(self, rollout_num, c=1, temperature=1, use_visits=False, v_function=None):
        MCTSPolicy.__init__(self, rollout_num, c, temperature, use_visits)
        TabularTreePolicy.__init__(self, v_function)


class TabularPUCTPolicy(PUCTPolicy, TabularTreePolicy):

    def __init__(self, rollout_count, c=1, temperature=1, use_visits=False, pi_function=None, v_function=None):
        PUCTPolicy.__init__(self, rollout_count, c, temperature, use_visits, pi_function)
        TabularTreePolicy.__init__(self, v_function)
