import random

from contexts import Context


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


class EpsilonGreedyTabularQPolicy(EpsilonGreedyPolicy, TabularQPolicy):

    def __init__(self, epsilon, q_function=None):
        EpsilonGreedyPolicy.__init__(self, epsilon)
        TabularQPolicy.__init__(self, q_function)