import unittest

import gym

import gym_socks.envs

from gym_socks.envs.sample import random_initial_conditions
from gym_socks.envs.sample import uniform_initial_conditions

from gym_socks.envs.sample import sample
from gym_socks.envs.sample import sample_trajectories

from gym_socks.envs.policy import ZeroPolicy

import numpy as np

system_list = [
    gym_socks.envs.NDIntegratorEnv(1),
    gym_socks.envs.NDIntegratorEnv(2),
    gym_socks.envs.NDIntegratorEnv(3),
    gym_socks.envs.NDIntegratorEnv(4),
    gym_socks.envs.StochasticNDIntegratorEnv(1),
    gym_socks.envs.StochasticNDIntegratorEnv(2),
    gym_socks.envs.StochasticNDIntegratorEnv(3),
    gym_socks.envs.StochasticNDIntegratorEnv(4),
    gym_socks.envs.NDPointMassEnv(1),
    gym_socks.envs.NDPointMassEnv(2),
    gym_socks.envs.NDPointMassEnv(3),
    gym_socks.envs.NDPointMassEnv(4),
    gym_socks.envs.StochasticNDPointMassEnv(1),
    gym_socks.envs.StochasticNDPointMassEnv(2),
    gym_socks.envs.StochasticNDPointMassEnv(3),
    gym_socks.envs.StochasticNDPointMassEnv(4),
    gym_socks.envs.NonholonomicVehicleEnv(),
    gym_socks.envs.StochasticNonholonomicVehicleEnv(),
    gym_socks.envs.CWH4DEnv(),
    gym_socks.envs.CWH6DEnv(),
    gym_socks.envs.StochasticCWH4DEnv(),
    gym_socks.envs.StochasticCWH6DEnv(),
    gym_socks.envs.QuadrotorEnv(),
    gym_socks.envs.StochasticQuadrotorEnv(),
]


class TestEnvironmentsRun(unittest.TestCase):
    def test_envs_run(cls):
        """Assert that environments can run."""

        for env in system_list:
            with cls.subTest(msg=f"Testing with {type(env)}."):

                try:
                    obs = env.reset()

                    for i in range(env.num_time_steps):

                        # get action
                        action = env.action_space.sample()

                        # apply action
                        obs, reward, done, _ = env.step(action)

                except Exception as e:
                    cls.fail(f"Simulating system {type(env)} raised an exception.")


class DummyDynamicalSystem(gym_socks.envs.dynamical_system.DynamicalSystem):
    """Dummy dynamical system class used for testing."""

    def __init__(self, *args, **kwargs):
        """Initialize the system."""
        super().__init__(*args, **kwargs)

    def dynamics(self, t, x, u):
        """Dummy dynamics for the system."""
        super().dynamics(t, x, u)


class TestDynamicalSystem(unittest.TestCase):
    """Dynamical system tests."""

    def setUp(cls):
        cls.system = DummyDynamicalSystem(
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            action_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            seed=1,
        )

    def test_should_fail_without_spaces(cls):
        """Test that system throws an error if spaces are not specified."""
        with cls.assertRaises(ValueError):
            system = DummyDynamicalSystem(observation_space=None, action_space=None)

        with cls.assertRaises(ValueError):
            system = DummyDynamicalSystem(
                observation_space=gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                action_space=None,
            )

        with cls.assertRaises(ValueError):
            system = DummyDynamicalSystem(
                observation_space=None,
                action_space=gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
            )

    def test_system_warns_time_step(cls):
        """
        Test that system throws a warning if sampling time is greater than time horizon.
        """

        with cls.assertWarns(Warning):
            cls.system.time_horizon = 1
            cls.system.sampling_time = 2

        with cls.assertWarns(Warning):
            cls.system.sampling_time = 1
            cls.system.time_horizon = 0.1

    def test_system_num_time_steps(cls):
        """System returns correct number of time steps."""

        cls.system.time_horizon = 1
        cls.system.sampling_time = 0.1

        # Note that this is not an error, and has to do with floating point precision.
        # See: https://docs.python.org/3/tutorial/floatingpoint.html
        cls.assertEqual(cls.system.num_time_steps, 9)
        cls.assertEqual(cls.system.time_horizon, 1)
        cls.assertEqual(cls.system.sampling_time, 0.1)

        cls.system.time_horizon = 10.0
        cls.system.sampling_time = 1.0

        # Note that this is not an error, and has to do with floating point precision.
        # See: https://docs.python.org/3/tutorial/floatingpoint.html
        cls.assertEqual(cls.system.num_time_steps, 10)
        cls.assertEqual(cls.system.time_horizon, 10)
        cls.assertEqual(cls.system.sampling_time, 1.0)

        cls.system.time_horizon = 5
        cls.system.sampling_time = 1

        cls.assertEqual(cls.system.num_time_steps, 5)
        cls.assertEqual(cls.system.time_horizon, 5)
        cls.assertEqual(cls.system.sampling_time, 1)

    def test_dims(cls):
        """State and action dims should match spaces."""
        cls.assertEqual(cls.system.state_dim, (1,))
        cls.assertEqual(cls.system.action_dim, (1,))

        cls.system.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

        cls.assertEqual(cls.system.state_dim, (2,))

    def test_cannot_change_dims(cls):
        """Cannot change the state and action dims."""

        with cls.assertRaises(AttributeError):
            cls.system.state_dim = 1

        with cls.assertRaises(AttributeError):
            cls.system.action_dim = 1

    def test_reset_returns_valid_state(cls):
        """Reset should return a valid state."""
        cls.system.reset()
        cls.assertTrue(cls.system.observation_space.contains(cls.system.state))

    def test_default_dynamics_not_implemented(cls):

        with cls.assertRaises(NotImplementedError):
            state = cls.system.reset()
            action = cls.system.action_space.sample()
            cls.system.dynamics(0, state, action)

    def test_default_close_not_implemented(cls):

        with cls.assertRaises(NotImplementedError):
            cls.system.close()

    def test_default_render_not_implemented(cls):

        with cls.assertRaises(NotImplementedError):
            cls.system.render()


class DummyStochasticDynamicalSystem(
    gym_socks.envs.dynamical_system.StochasticMixin,
    gym_socks.envs.dynamical_system.DynamicalSystem,
):
    """Dummy stochastic dynamical system class used for testing."""

    def __init__(self, *args, **kwargs):
        """Initialize the system."""
        super().__init__(*args, **kwargs)

    def dynamics(self, t, x, u):
        """Dummy dynamics for the system."""
        super().dynamics(t, x, u)


class TestStochasticDynamicalSystem(unittest.TestCase):
    """Stochastic dynamical system tests."""

    def setUp(cls):
        cls.system = DummyStochasticDynamicalSystem(
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            action_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            disturbance_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            seed=1,
        )

    def test_should_fail_without_spaces(cls):
        """Test that system throws an error without disturbance space."""
        with cls.assertRaises(ValueError):
            system = DummyStochasticDynamicalSystem(
                observation_space=gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                action_space=gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                disturbance_space=None,
            )

    def test_default_dynamics_not_implemented(cls):

        with cls.assertRaises(NotImplementedError):
            state = cls.system.reset()
            action = cls.system.action_space.sample()
            cls.system.dynamics(0, state, action)


class Test4DCWHSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym_socks.envs.cwh.CWH4DEnv()

    def test_known_trajectory(cls):
        """Test against specific known trajectory. Sanity check."""

        env = cls.env

        env.state = np.array([-0.1, -0.1, 0, 0])
        action = np.array([0, 0])

        trajectory = []
        trajectory.append(env.state)

        for i in range(3):
            obs, reward, done, _ = env.step(action)
            trajectory.append(obs)

        trajectory = np.array(trajectory)

        n = env.angular_velocity
        nt = n * env.sampling_time
        sin_nt = np.sin(nt)
        cos_nt = np.cos(nt)

        A = [
            [4 - 3 * cos_nt, 0, (1 / n) * sin_nt, (2 / n) * (1 - cos_nt)],
            [
                6 * (sin_nt - nt),
                1,
                -(2 / n) * (1 - cos_nt),
                (1 / n) * (4 * sin_nt - 3 * nt),
            ],
            [3 * n * sin_nt, 0, cos_nt, 2 * sin_nt],
            [-6 * n * (1 - cos_nt), 0, -2 * sin_nt, 4 * cos_nt - 3],
        ]

        state = np.array([-0.1, -0.1, 0, 0])

        groundTruth = []
        groundTruth.append(state)

        for i in range(3):
            state = np.matmul(A, state.T)
            groundTruth.append(state)

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(trajectory, groundTruth),
            "Generated trajectory should match known trajectory generated using dynamics.",
        )

        state = np.array([-0.1, 0.1, 0, 0])

        falseTrajectory = []
        falseTrajectory.append(state)

        for i in range(3):
            state = np.matmul(A, state.T)
            falseTrajectory.append(state)

        cls.assertFalse(
            np.allclose(trajectory, falseTrajectory),
            "Generated trajectory should not match known false trajectory generated using dynamics.",
        )


class Test6DCWHSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym_socks.envs.cwh.CWH6DEnv()

    def test_known_trajectory(cls):
        """Test against specific known trajectory. Sanity check."""

        env = cls.env

        env.state = np.array([-0.1, -0.1, 0, 0, 0, 0])
        action = np.array([0, 0, 0])

        trajectory = []
        trajectory.append(env.state)

        for i in range(3):
            obs, reward, done, _ = env.step(action)
            trajectory.append(obs)

        trajectory = np.array(trajectory)

        n = env.angular_velocity
        nt = n * env.sampling_time
        sin_nt = np.sin(nt)
        cos_nt = np.cos(nt)

        A = [
            [4 - 3 * cos_nt, 0, 0, (1 / n) * sin_nt, (2 / n) * (1 - cos_nt), 0],
            [
                6 * (sin_nt - nt),
                1,
                0,
                -(2 / n) * (1 - cos_nt),
                (1 / n) * (4 * sin_nt - 3 * nt),
                0,
            ],
            [0, 0, cos_nt, 0, 0, (1 / n) * sin_nt],
            [3 * n * sin_nt, 0, 0, cos_nt, 2 * sin_nt, 0],
            [-6 * n * (1 - cos_nt), 0, 0, -2 * sin_nt, 4 * cos_nt - 3, 0],
            [0, 0, -n * sin_nt, 0, 0, cos_nt],
        ]

        state = np.array([-0.1, -0.1, 0, 0, 0, 0])

        groundTruth = []
        groundTruth.append(state)

        for i in range(3):
            state = np.matmul(A, state.T)
            groundTruth.append(state)

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(trajectory, groundTruth),
            "Generated trajectory should match known trajectory generated using dynamics.",
        )

        state = np.array([-0.1, 0.1, 0, 0, 0, 0])

        falseTrajectory = []
        falseTrajectory.append(state)

        for i in range(3):
            state = np.matmul(A, state.T)
            falseTrajectory.append(state)

        cls.assertFalse(
            np.allclose(trajectory, falseTrajectory),
            "Generated trajectory should not match known false trajectory generated using dynamics.",
        )


class TestIntegratorSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym_socks.envs.NDIntegratorEnv(2)

    def test_known_trajectory(cls):
        """Test against specific known trajectory. Sanity check."""

        env = cls.env

        env.state = np.array([0.1, 0.1])
        action = np.array([0])

        trajectory = []
        trajectory.append(env.state)

        for i in range(cls.env.num_time_steps):
            obs, reward, done, _ = env.step(action)
            trajectory.append(obs)

        trajectory = np.array(trajectory)

        groundTruth = np.array(
            [
                [0.100, 0.1],
                [0.125, 0.1],
                [0.150, 0.1],
                [0.175, 0.1],
                [0.200, 0.1],
                [0.225, 0.1],
                [0.250, 0.1],
                [0.275, 0.1],
                [0.300, 0.1],
                [0.325, 0.1],
                [0.350, 0.1],
                [0.375, 0.1],
                [0.400, 0.1],
                [0.425, 0.1],
                [0.450, 0.1],
                [0.475, 0.1],
                [0.500, 0.1],
            ]
        )

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(trajectory, groundTruth),
            "Generated trajectory should match known trajectory generated using dynamics.",
        )


class TestNonholonomicSystem(unittest.TestCase):
    def test_corrects_angle(cls):
        system = gym_socks.envs.NonholonomicVehicleEnv()
        system.state = np.array([0, 0, 4 * np.pi])
        action = [0, 0]
        obs, reward, done, _ = system.step(action)

        cls.assertTrue((system.state[2] <= 2 * np.pi) & (system.state[2] >= -2 * np.pi))


class TestStochasticNonholonomicSystem(unittest.TestCase):
    def test_corrects_angle(cls):
        system = gym_socks.envs.StochasticNonholonomicVehicleEnv()
        system.disturbance_space = gym.spaces.Box(
            low=0, high=0, shape=(3,), dtype=np.float32
        )

        system.state = np.array([0, 0, 4 * np.pi])
        action = [0, 0]
        obs, reward, done, _ = system.step(action)

        cls.assertTrue((system.state[2] <= 2 * np.pi) & (system.state[2] >= -2 * np.pi))


class DummyPolicy(gym_socks.envs.policy.BasePolicy):
    def __call__(self):
        super().__call__()


class TestBasePolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = DummyPolicy()

    def test_default_policy_not_implemented(cls):

        with cls.assertRaises(NotImplementedError):
            cls.policy()


class TestConstantPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = DummyDynamicalSystem(
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            action_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            seed=1,
        )

    def test_constant_policy_returns_constants(cls):
        policy = gym_socks.envs.policy.ConstantPolicy(system=cls.system, constant=1)
        cls.assertEqual(policy(), [1])

        policy = gym_socks.envs.policy.ConstantPolicy(system=cls.system, constant=5.0)
        cls.assertEqual(policy(), [5.0])


class TestGenerateInitialConditions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = system_list

    def test_incorrect_sample_space_dimensionality(cls):
        """Test for failure if sample_space is incorrect dimensionality."""

        for env in cls.envs:
            with cls.subTest(msg=f"Testing with {type(env)}."):

                sample_space = gym.spaces.Box(
                    low=-1,
                    high=1,
                    shape=(env.observation_space.shape[0] + 1,),
                    dtype=np.float32,
                )

                cls.assertNotEqual(env.observation_space.shape, sample_space.shape)
                with cls.assertRaises(AssertionError) as exception_context:
                    ic = random_initial_conditions(
                        sample_space=sample_space, system=env, n=5
                    )

                cls.assertNotEqual(env.observation_space.shape, sample_space.shape)
                with cls.assertRaises(AssertionError) as exception_context:
                    ic = uniform_initial_conditions(
                        sample_space=sample_space, system=env, n=5
                    )


class TestGenerateSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = system_list

    def test_sample(cls):
        """Assert that sample generates a list of state observations."""

        for env in cls.envs:
            with cls.subTest(msg=f"Testing with {type(env)}."):

                obs_shape = env.observation_space.shape

                sample_space = gym.spaces.Box(
                    low=-1,
                    high=1,
                    shape=obs_shape,
                    dtype=np.float32,
                )

                initial_conditions = (
                    [[1] * obs_shape[0]],
                    [[1] * obs_shape[0], [2] * obs_shape[0]],
                    random_initial_conditions(system=env, sample_space=sample_space),
                    uniform_initial_conditions(system=env, sample_space=sample_space),
                )

                for ic in initial_conditions:
                    with cls.subTest(msg=f"Testing with different ICs."):

                        S, U = sample(system=env, initial_conditions=ic)

                        cls.assertIsInstance(S, np.ndarray, "Should be an ndarray.")
                        cls.assertEqual(
                            np.shape(S),
                            (np.array(ic).shape[0], 2, obs_shape[0]),
                            "Should return the correct dimensions.",
                        )

                        cls.assertIsInstance(U, np.ndarray, "Should be an ndarray.")
                        cls.assertEqual(
                            np.shape(U),
                            (np.array(ic).shape[0], 1, env.action_space.shape[0]),
                            "Should return the correct dimensions.",
                        )

    def test_known_sample(cls):
        """Test against specific known sample."""

        env = gym_socks.envs.integrator.NDIntegratorEnv(2)

        policy = ZeroPolicy(env)

        S, U = sample(
            system=env,
            initial_conditions=[[0.1, 0.1], [0.125, 0.1]],
            policy=policy,
        )

        groundTruth = np.array(
            [
                [
                    [0.100, 0.1],
                    [0.125, 0.1],
                ],
                [
                    [0.125, 0.1],
                    [0.150, 0.1],
                ],
            ]
        )

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(np.allclose(S, groundTruth))


class TestGenerateSampleTrajectories(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = system_list

    def test_sample_trajectories(cls):
        """Assert that generate_state_trajectory generates state observations."""

        for env in cls.envs:
            with cls.subTest(msg=f"Testing with {type(env)}."):

                obs_shape = env.observation_space.shape

                sample_space = gym.spaces.Box(
                    low=-1,
                    high=1,
                    shape=obs_shape,
                    dtype=np.float32,
                )

                initial_conditions = (
                    [[1] * obs_shape[0]],
                    [[1] * obs_shape[0], [2] * obs_shape[0]],
                    random_initial_conditions(system=env, sample_space=sample_space),
                    uniform_initial_conditions(system=env, sample_space=sample_space),
                )

                for ic in initial_conditions:
                    with cls.subTest(msg=f"Testing with different ICs."):

                        S, U = sample_trajectories(system=env, initial_conditions=ic)

                        len = env.num_time_steps + 1

                        cls.assertIsInstance(S, np.ndarray, "Should be an ndarray.")
                        cls.assertEqual(
                            np.shape(S),
                            (np.array(ic).shape[0], len, obs_shape[0]),
                            "Should return the correct dimensions.",
                        )

                        cls.assertIsInstance(U, np.ndarray, "Should be an ndarray.")
                        cls.assertEqual(
                            np.shape(U),
                            (np.array(ic).shape[0], len - 1, env.action_space.shape[0]),
                            "Should return the correct dimensions.",
                        )

    def test_known_trajectory(cls):
        """Test against specific known trajectory."""

        env = gym_socks.envs.integrator.NDIntegratorEnv(2)

        policy = ZeroPolicy(env)

        S, U = sample_trajectories(
            system=env, initial_conditions=[[0.1, 0.1]], policy=policy
        )

        groundTruth = np.array(
            [
                [
                    [0.100, 0.1],
                    [0.125, 0.1],
                    [0.150, 0.1],
                    [0.175, 0.1],
                    [0.200, 0.1],
                    [0.225, 0.1],
                    [0.250, 0.1],
                    [0.275, 0.1],
                    [0.300, 0.1],
                    [0.325, 0.1],
                    [0.350, 0.1],
                    [0.375, 0.1],
                    [0.400, 0.1],
                    [0.425, 0.1],
                    [0.450, 0.1],
                    [0.475, 0.1],
                    [0.500, 0.1],
                ]
            ]
        )

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(np.allclose(S, groundTruth))
