import random
import math

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
            q_values = [q_values[action] for action in context.actions]
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
            values.append(value)
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


class GreedyPolicy(ScorePolicy):

    def __call__(self, context: Context):
        scores = self.scores(context)
        best_score = None
        best_actions = None
        action_scores = dict()
        for action, score in zip(context.actions, scores):
            action_scores[action] = score
            if best_score is None or best_score * context.move < score * context.move:
                best_score = score
                best_actions = [action]
            elif best_score == score:
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
        best_score = None
        action_scores = dict()
        for action, score in zip(context.actions, scores):
            action_scores[action] = score
            if best_score is None or best_score * context.move < score * context.move:
                best_score = score
        weights = [math.exp(context.move * (score - best_score) / self.temperature) for score in scores]
        stat_sum = sum(weights)
        action_proba = [weight / stat_sum for weight in weights]
        policy_action = random.choices(context.actions, action_proba)[0]
        return policy_action, {
            'policy': 'boltzmann',
            'scores': action_scores,
            'probability': {action: proba for action, proba in zip(context.actions, action_proba)}
        }


class GreedyTabularQPolicy(GreedyPolicy, TabularQPolicy):

    def __init__(self, v_function=None):
        TabularQPolicy.__init__(self, v_function)


class GreedyTabularVPolicy(GreedyPolicy, TabularVPolicy):

    def __init__(self, v_function=None):
        TabularVPolicy.__init__(self, v_function)


class GreedyTabularPiPolicy(GreedyPolicy, TabularPiPolicy):

    def __init__(self, pi_policy=None):
        TabularPiPolicy.__init__(self, pi_policy)


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


class MCTSPolicy(Policy):

    def __init__(self, rollout_count, c, temperature, use_visits=False, default_policy=None):
        self.rollout_count = rollout_count
        self.c = c
        self.temperature = temperature
        self.use_visits = use_visits
        self.default_policy = default_policy or RandomPolicy()

    def expand(self, context: ContextTree):
        while not context.done:
            action, _ = self.default_policy(context)
            context = context.of(action)
        return context.reward

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
            best_bound = None
            best_actions = None
            for action in context.actions:
                child = context.children[action]
                child_bound = child.value + self.c * context.move * math.sqrt(math.log(context.visits) / child.visits)
                if best_bound is None or child_bound * context.move > best_bound * context.move:
                    best_bound = child_bound
                    best_actions = [action]
                elif child_bound == best_bound:
                    best_actions.append(action)
            selected_action = random.choice(best_actions)
            context = context(selected_action)

    @staticmethod
    def backward(context, history, value):
        for action in history:
            child = context(action)
            child.value += (value - child.value) / child.visits
            context = child

    def __call__(self, context: ContextTree):
        for _ in range(self.rollout_count):
            history, value = self.select(context)
            MCTSPolicy.backward(context, history[len(context.history):], value)
        actions = list()
        action_values = list()
        action_visits = list()
        best_value = None
        max_visits = None
        for action in context.actions:
            child: ContextTree = context.children[action]
            if child is not None:
                actions.append(action)
                action_values.append(child.value)
                action_visits.append(child.visits)
                if self.use_visits:
                    if max_visits is None or max_visits < child.visits:
                        max_visits = child.visits
                else:
                    if best_value is None or best_value * context.move < child.value * context.move:
                        best_value = child.value
        if self.use_visits:
            weights = [(visits / max_visits) ** (1 / self.temperature) for visits in action_visits]
        else:
            weights = [math.exp(context.move * (value - best_value) / self.temperature) for value in action_values]
        stat_sum = sum(weights)
        action_proba = [weight / stat_sum for weight in weights]
        policy_action = random.choices(actions, action_proba)[0]
        return policy_action, {
            'policy': 'mcts',
            'values': {action: value for action, value in zip(actions, action_values)},
            'visits': {action: visits for action, visits in zip(actions, action_visits)},
            'probability': {action: proba for action, proba in zip(actions, action_proba)}
        }
