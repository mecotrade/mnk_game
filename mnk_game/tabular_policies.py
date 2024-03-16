import random

import policies
from contexts import Context, ContextPredictor


class TabularQPolicy(policies.ScorePolicy):

    def __init__(self, q_function=None, max_init_q=0.01):
        self.q_function = q_function or dict()
        self.max_init_q = max_init_q

    def scores(self, context: Context):
        q_values = self.q_function.setdefault(context.board, self.init(context.num_actions()))
        return [q_values[action] * context.move for action in context.actions]

    def init(self, num_actions):
        return [(2 * random.random() - 1) * self.max_init_q for _ in range(num_actions)]


class TabularVPolicy(policies.ScorePolicy):

    def __init__(self, v_function=None, max_init_value=0.01):
        self.v_function = v_function or dict()
        self.max_init_value = max_init_value

    def scores(self, context: Context):
        values = list()
        for action in context.actions:
            virtual_board = context.apply(action)
            value = self.v_function.setdefault(virtual_board, self.init())
            values.append(value * context.move)
        return values

    def init(self):
        return (2 * random.random() - 1) * self.max_init_value


class TabularPiPolicy(policies.ScorePolicy):

    def __init__(self, pi_function=None):
        self.pi_function = pi_function or dict()

    def scores(self, context: Context):
        _, scores = self.pi_function.setdefault(context.board, self.uniform(context.num_actions()))
        return [scores[action] for action in context.actions]

    @staticmethod
    def uniform(num_actions: int):
        return [1 / num_actions] * num_actions, [0.] * num_actions


class GreedyTabularQPolicy(policies.GreedyPolicy, TabularQPolicy):

    def __init__(self, v_function=None):
        TabularQPolicy.__init__(self, v_function)


class GreedyTabularVPolicy(policies.GreedyPolicy, TabularVPolicy):

    def __init__(self, v_function=None):
        TabularVPolicy.__init__(self, v_function)


class GreedyTabularPiPolicy(policies.GreedyPolicy, TabularPiPolicy):

    def __init__(self, pi_function=None):
        TabularPiPolicy.__init__(self, pi_function)


class EpsilonGreedyTabularQPolicy(policies.EpsilonGreedyPolicy, TabularQPolicy):

    def __init__(self, epsilon, q_function=None):
        policies.EpsilonGreedyPolicy.__init__(self, epsilon)
        TabularQPolicy.__init__(self, q_function)


class BoltzmannTabularVPolicy(policies.BoltzmannPolicy, TabularVPolicy):

    def __init__(self, temperature=1., v_function=None):
        policies.BoltzmannPolicy.__init__(self, temperature)
        TabularVPolicy.__init__(self, v_function)


class BoltzmannTabularQPolicy(policies.BoltzmannPolicy, TabularQPolicy):

    def __init__(self, temperature=1., q_function=None):
        policies.BoltzmannPolicy.__init__(self, temperature)
        TabularQPolicy.__init__(self, q_function)


class BoltzmannTabularPiPolicy(policies.BoltzmannPolicy, TabularPiPolicy):

    def __init__(self, temperature=1., pi_function=None):
        policies.BoltzmannPolicy.__init__(self, temperature)
        TabularPiPolicy.__init__(self, pi_function)


class TabularPUCTPolicy(policies.PUCTPolicy):

    def __init__(self, rollout_count, c=1, temperature=1, use_visits=False, pi_function=None):
        super().__init__(rollout_count, c, temperature, use_visits)
        self.pi_function = pi_function or dict()

    def expand(self, context: ContextPredictor):
        pi, _ = self.pi_function.setdefault(context.board, TabularPiPolicy.uniform(context.num_actions()))
        context.predictor = pi
        return super().expand(context)


class TabularPUCTDefaultPolicy(TabularPUCTPolicy, policies.DefaultTreePolicy):

    def __init__(self, rollout_count, c=1, temperature=1, use_visits=False, pi_function=None, default_policy=None):
        TabularPUCTPolicy.__init__(self, rollout_count, c, temperature, use_visits, pi_function)
        policies.DefaultTreePolicy.__init__(self, default_policy)


class TabularVTreePolicy(policies.TreePolicy):

    def __init__(self, v_function=None, max_init_value=0.01):
        self.v_function = v_function or dict()
        self.max_init_value = max_init_value

    def init(self):
        return (2 * random.random() - 1) * self.max_init_value

    def expand(self, context):
        return context.reward if context.done else self.v_function.setdefault(context.board, self.init())


class TabularVUCTPolicy(policies.MCTSPolicy, TabularVTreePolicy):

    def __init__(self, rollout_num, c=1, temperature=1, use_visits=False, v_function=None):
        policies.MCTSPolicy.__init__(self, rollout_num, c, temperature, use_visits)
        TabularVTreePolicy.__init__(self, v_function)


class TabularVTabularPUCTPolicy(TabularPUCTPolicy, TabularVTreePolicy):

    def __init__(self, rollout_count, c=1, temperature=1, use_visits=False, pi_function=None, v_function=None):
        TabularPUCTPolicy.__init__(self, rollout_count, c, temperature, use_visits, pi_function)
        TabularVTreePolicy.__init__(self, v_function)
