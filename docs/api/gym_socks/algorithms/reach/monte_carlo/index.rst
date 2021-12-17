:py:mod:`~gym_socks.algorithms.reach.monte_carlo`
=================================================

.. py:module:: gym_socks.algorithms.reach.monte_carlo

.. autoapi-nested-parse::

   Stochastic reachability using Monte-Carlo.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.algorithms.reach.monte_carlo.MonteCarloSR



Functions
~~~~~~~~~

.. autoapisummary::

   gym_socks.algorithms.reach.monte_carlo.monte_carlo_sr



.. py:function:: monte_carlo_sr(env, policy, T, num_iterations = None, time_horizon = None, constraint_tube = None, target_tube = None, problem = 'THT', verbose = False)

   Stochastic reachability using Monte-Carlo.

   Computes an approximation of the safety probabilities of the stochastic reachability
   problem using Monte-Carlo methods.

   :param env: The dynamical system model.
   :param policy: The policy applied to the system during sampling.
   :param T: Points to estimate the safety probabilities at. Should be in the form of a
             2D-array, where each row indicates a point.
   :param num_iterations: Number of Monte-Carlo iterations.
   :param constraint_tube: List of spaces or constraint functions. Must be the same
                           length as `num_steps`.
   :param target_tube: List of spaces or target functions. Must be the same length as
                       `num_steps`.
   :param problem: One of `{"THT", "FHT"}`. `"THT"` specifies the terminal-hitting time
                   problem and `"FHT"` specifies the first-hitting time problem.
   :param verbose: Boolean flag to indicate verbose output.


.. py:class:: MonteCarloSR(num_iterations = None, time_horizon = None, constraint_tube = None, target_tube = None, problem = 'THT', verbose = False, *args, **kwargs)

   Bases: :py:obj:`gym_socks.algorithms.algorithm.AlgorithmInterface`

   Stochastic reachability using Monte-Carlo.

   Computes an approximation of the safety probabilities of the stochastic reachability
   problem using Monte-Carlo methods.

   :param num_iterations: Number of Monte-Carlo iterations.
   :param time_horizon: Time horizon of the algorithm.
   :param constraint_tube: List of spaces or constraint functions. Must be the same
                           length as `num_steps`.
   :param target_tube: List of spaces or target functions. Must be the same length as
                       `num_steps`.
   :param problem: One of `{"THT", "FHT"}`. `"THT"` specifies the terminal-hitting time
                   problem and `"FHT"` specifies the first-hitting time problem.
   :param verbose: Boolean flag to indicate verbose output.

   .. py:method:: fit(self)
      :abstractmethod:


   .. py:method:: predict(self)
      :abstractmethod:


   .. py:method:: fit_predict(self, env, policy, T)

      Run the algorithm.

      Computes the safety probabilities for the points provided. For each point in
      `T`, the algorithm computes a collection of trajectories using the point as the
      initial condition. Then, we can evaluate the indicator functions for each
      generated trajectory and the estimated safety probability is the sum of
      indicators divided by the number of trajectories.

      :param env: The dynamical system model. Needed to configure the sampling spaces.
      :param T: Points to estimate the safety probabilities at. Should be in the form of
                a 2D-array, where each row indicates a point.

      :returns: The safety probabilities corresponding to each point. The output is in the
                form of a 2D-array, where each row corresponds to the points in `T` and the
                number of columns corresponds to the number of time steps.
