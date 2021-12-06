import numpy as np

from sacred import Ingredient


cost_ingredient = Ingredient("cost")


@cost_ingredient.config
def cost_config():

    goal = [10, 0, 10, 0]


@cost_ingredient.capture
def make_cost_x(time_horizon):
    def _cost_fn(time, state):
        return np.zeros((np.shape(state)[0],))

    return _cost_fn


@cost_ingredient.capture
def make_cost_u(time_horizon):
    def _cost_fn(time, action):
        action = np.reshape(action, (-1, time_horizon, 2))
        result = np.linalg.norm(action, ord=1, axis=2)
        return np.sum(result, axis=1)

    return _cost_fn
