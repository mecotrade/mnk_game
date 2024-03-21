import os.path

import numpy as np
import torch

import policies
import models
from contexts import Context, ContextPredictor


class NetPolicy(policies.Policy):

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        if not os.path.exists(path):
            torch.save(self.model.state_dict(), path)


class NetQPolicy(policies.ScorePolicy, NetPolicy):

    def __init__(self, model: models.QModel):
        NetPolicy.__init__(self, model)

    def scores(self, context: Context):
        q_values = self.model(torch
                              .from_numpy(context.features()[np.newaxis, :])
                              .permute((0, 3, 1, 2))).detach().numpy()[0]
        return [q_values[action] * context.move for action in context.actions]


class NetVPolicy(policies.ScorePolicy, NetPolicy):
    def __init__(self, model: models.VModel):
        NetPolicy.__init__(self, model)

    def scores(self, context: Context):
        features = list()
        for action in context.actions:
            virtual_context = context(action)
            features.append(virtual_context.features())
        values = context.move * self.model(torch.from_numpy(np.stack(features)).permute(0, 3, 1, 2)).detach().numpy().reshape([-1])
        return values.tolist()


class GreedyNetVPolicy(policies.GreedyPolicy, NetVPolicy):
    def __init__(self, model):
        NetVPolicy.__init__(self, model)


class EpsilonGreedyNetVPolicy(policies.EpsilonGreedyPolicy, NetVPolicy):
    def __init__(self, model, epsilon):
        policies.EpsilonGreedyPolicy.__init__(self, epsilon)
        NetVPolicy.__init__(self, model)


class GreedyNetQPolicy(policies.GreedyPolicy, NetQPolicy):
    def __init__(self, model):
        NetQPolicy.__init__(self, model)


class EpsilonGreedyNetQPolicy(policies.EpsilonGreedyPolicy, NetQPolicy):
    def __init__(self, model, epsilon):
        policies.EpsilonGreedyPolicy.__init__(self, epsilon)
        NetQPolicy.__init__(self, model)


class BoltzmannNetQPolicy(policies.BoltzmannPolicy, NetQPolicy):
    def __init__(self, model, temperature=1.):
        policies.BoltzmannPolicy.__init__(self, temperature)
        NetQPolicy.__init__(self, model)


class BoltzmannNetVPolicy(policies.BoltzmannPolicy, NetVPolicy):
    def __init__(self, model, temperature=1.):
        policies.BoltzmannPolicy.__init__(self, temperature)
        NetVPolicy.__init__(self, model)


class NetPUCTPolicy(policies.PUCTPolicy, NetPolicy):
    def __init__(self, model, rollout_count, c=1, temperature=1, use_visits=False):
        policies.PUCTPolicy.__init__(self, rollout_count, c, temperature, use_visits)
        NetPolicy.__init__(self, model)

    def expand(self, context: ContextPredictor):
        logits = self.model(torch.from_numpy(context.features()[np.newaxis, :]).permute((0, 3, 1, 2)))
        context.predictor = torch.softmax(logits, dim=1).detach().numpy().reshape([-1]).tolist()
        return super().expand(context)


class NetPUCTDefaultPolicy(NetPUCTPolicy, policies.DefaultTreePolicy):
    def __init__(self, model, rollout_count, c=1, temperature=1, use_visits=False, default_policy=None):
        NetPUCTPolicy.__init__(self, model, rollout_count, c, temperature, use_visits)
        policies.DefaultTreePolicy.__init__(self, default_policy)


class AlphaPolicy(policies.PUCTPolicy, NetPolicy):

    def __init__(self, model: models.AlphaModel, rollout_count, c=1, temperature=1, use_visits=False):
        policies.PUCTPolicy.__init__(self, rollout_count, c, temperature, use_visits)
        NetPolicy.__init__(self, model)

    def expand(self, context: ContextPredictor):
        logits, v = self.model(torch.from_numpy(context.features()[np.newaxis, :]).permute((0, 3, 1, 2)))
        context.predictor = torch.softmax(logits, dim=1).detach().numpy().reshape([-1]).tolist()
        value = v.detach().numpy().squeeze()
        return value




