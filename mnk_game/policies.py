import random
import math

from contexts import Context, ContextTree, ContextPredictor


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


class GreedyPolicy(ScorePolicy):

    def __call__(self, context: Context):
        scores = self.scores(context)
        max_score = None
        policy_action = None
        action_scores = dict()
        for action, score in zip(context.actions, scores):
            action_scores[action] = score
            if max_score is None or max_score < score:
                max_score = score
                policy_action = action
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
            selected_action = None
            for action in context.actions:
                child = context.children[action]
                child_bound = (context.move / child.move) * child.value + self.c * math.sqrt(math.log(context.visits) / child.visits)
                if max_bound is None or child_bound > max_bound:
                    max_bound = child_bound
                    selected_action = action
            context = context(selected_action)

    @staticmethod
    def backward(context, history, reward):
        for action in history:
            child = context(action)
            child.value += (reward * child.move - child.value) / child.visits
            context = child

    def __call__(self, context: ContextTree):
        if context.visits == 0:
            self.expand(context)
        for _ in range(self.rollout_count):
            history, reward = self.select(context)
            self.backward(context, history[len(context.history):], reward)
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

    def __init__(self, rollout_count, c=1, temperature=1, use_visits=False):
        super().__init__(rollout_count, c, temperature, use_visits)

    def select(self, context: ContextPredictor):
        while True:
            context.visits += 1
            if context.done:
                return context.history, context.reward
            max_bound = None
            selected_action = None
            for action in context.actions:
                child = context(action)
                child_bound = (context.move / child.move) * child.value + self.c * context.predictor[action] * math.sqrt(context.visits) / (child.visits + 1)
                if max_bound is None or child_bound > max_bound:
                    max_bound = child_bound
                    selected_action = action
            context = context(selected_action)
            if context.visits == 0:
                context.visits += 1
                return context.history, self.expand(context)

    def __call__(self, context: ContextPredictor):
        action, info = super().__call__(context)
        info['policy'] = 'puct'
        info['predictors'] = {action: context.predictor[action] for action in context.actions}
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

    def __init__(self, rollout_count, c=1, temperature=1, use_visits=False, default_policy=None):
        PUCTPolicy.__init__(self, rollout_count, c, temperature, use_visits)
        DefaultTreePolicy.__init__(self, default_policy)
