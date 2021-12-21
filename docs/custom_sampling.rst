Sampling
========

Templates
---------

.. tab-set::

    .. tab-item:: Uniform

        .. code-block:: python

            from gym_socks.sampling import sample_generator

            @sample_generator
            def custom_sampler(env, policy, sample_space):
                env.state = sample_space.sample()
                action = policy(state=state)
                next_state, *_ = env.step(action)
                yield (env.state, action, next_state)

    .. tab-item:: Grid

        .. code-block:: python

            from gym_socks.sampling import sample_generator

            @sample_generator
            def custom_sampler(env):

                state = env.state_space.sample()

                yield state

    .. tab-item:: Policy

        .. code-block:: python

            from gym_socks.sampling import sample_generator

            @sample_generator
            def custom_sampler(env):

                state = env.state_space.sample()

                yield state