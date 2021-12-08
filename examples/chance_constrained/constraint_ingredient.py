import numpy as np

from sacred import Ingredient

obstacle_ingredient = Ingredient("obstacle")


@obstacle_ingredient.config
def obstacle_config():

    obstacles = [
        {"A": [[-1, 1], [1, 0], [0, -1]], "b": [-0.7, 8, -2]},
        {"A": [[0, 1], [-1, 0], [1, -1]], "b": [7, -3, -0.7]},
    ]


constraint_ingredient = Ingredient("constraint", ingredients=[obstacle_ingredient])


@constraint_ingredient.config
def constraint_config():

    # Epsilon distance around X_goal.
    epsilon = 2.5


@constraint_ingredient.capture
def make_constraint_x(time_horizon, obstacle, epsilon):
    def _constraint_fn(time, state):
        state = np.reshape(state, (-1, time_horizon, 4))

        state_shape = np.shape(state)
        indicator = np.zeros((state_shape[0],), dtype=bool)

        # Obstacle constraints.
        for obs in obstacle["obstacles"]:

            Oi_A = np.array(obs["A"])
            Oi_b = np.array(obs["b"])

            for i in range(state_shape[0]):

                for j in range(time_horizon):

                    in_obstacle = True

                    for k in range(3):
                        h_ij = -np.array([Oi_A[k, 0], 0, Oi_A[k, 1], 0])
                        g_ij = -Oi_b[k]

                        if h_ij @ state[i, j, :] <= g_ij:
                            in_obstacle = False

                    indicator[i] = indicator[i] or in_obstacle

        # X_goal constraint
        dist = state[:, -1, [0, 2]] - np.array([10, 10])
        result = np.linalg.norm(dist, ord=2, axis=1)
        in_goal = result <= epsilon

        return ~indicator & in_goal
        # return ~indicator

    return _constraint_fn


@constraint_ingredient.capture
def make_constraint_u(time_horizon):
    def _constraint_fn(time, action):
        return np.zeros((np.shape(action)[0],))

    return _constraint_fn
