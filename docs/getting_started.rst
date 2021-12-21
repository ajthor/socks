Getting Started
===============

SOCKS is a suite of algorithms for stochastic optimal control using kernel methods.

It runs on top of `OpenAI Gym <https://gym.openai.com>`_, and comes with several classic
controls Gym environments. In addition, it can integrate with many pre-existing Gym
environments.

Examples
--------

.. grid:: 2
    :gutter: 1 1 2 3

    .. grid-item-card::
        :link: examples/reach/stoch_reach_maximal
        :link-type: doc

        **Maximal Stochastic Reachability**
        ^^^

        Compute a policy that maximizes the probability of remaining within a safe set
        and reaching a target set.

        +++
        :bdg-primary-line:`control`
        :bdg-primary-line:`reachability`

    .. grid-item-card::
        :link: examples/reach/forward_reach
        :link-type: doc

        **Forward Reachability**
        ^^^

        Compute a forward reachable set classifier.

        +++
        :bdg-primary-line:`reachability`

    .. grid-item-card::
        :link: examples/control/tracking
        :link-type: doc

        **Target Tracking Problem**
        ^^^

        Unconstrained stochastic optimal control.

        +++
        :bdg-primary-line:`control`

    .. grid-item-card::
        :link: examples/control/obstacle_avoid
        :link-type: doc

        **Obstacle Avoid**
        ^^^

        Constrained stochastic optimal control.

        +++
        :bdg-primary-line:`control`
