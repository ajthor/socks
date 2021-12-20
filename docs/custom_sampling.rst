Sampling
========

Templates
---------

.. tab:: Uniform

    .. code-block:: python

        from gym_socks.sampling import sample_generator

        @sample_generator
        def custom_sampler(env):

            state = env.state_space.sample()

            yield state

.. tab:: Grid

    .. code-block:: python

        from gym_socks.sampling import sample_generator

        @sample_generator
        def custom_sampler(env):

            state = env.state_space.sample()

            yield state

.. tab:: Policy

    .. code-block:: python

        from gym_socks.sampling import sample_generator

        @sample_generator
        def custom_sampler(env):

            state = env.state_space.sample()

            yield state