:py:mod:`~examples.ingredients.sample_ingredient`
=================================================

.. py:module:: examples.ingredients.sample_ingredient

.. autoapi-nested-parse::

   Sample ingredient.

   The sample ingredient is used when an experiment requires a sample from a system to be
   generated. It provides configuration variables used to set up the sample scheme, the
   shape of the sample space, as well as the sample size. It also provides functions used
   to generate the samples used in the experiments.

   The main configuration specifications for the sample are given by:

   * The need to specify different types of sampling schemes.
   * The need to specify the bounds of the sampling region.
   * The need to specify the "density" of the samples or, alternatively, the sample size.

   The sampling ingredient should then provide functions which:

   * Generate the appropriate sampling spaces.
   * Provide an interface to the sampling functions that provide a single point of entry.

   .. rubric:: Notes

   The main problem is that sacred does not allow for "substituting" different
   configurations depending on other configuration choices. For instance, if a user
   specifies a certain sampling scheme, there is no explicit method to modify the other
   variables required, since the ingredients are not dynamic.

   Ideally, the user would be able to specify a certain sampling scheme, and an
   ingredient could be loaded and added to the experiment "dynamically", injecting new
   configuration variables that need to be specified for the particular sampling
   scheme and providing a consistent "interface" that can be used regardless of the
   chosen sampling scheme.

   However, sacred *does* allow for configuration "hooks" which means we can
   dynamically add configuration variables at runtime. Additionally, dictionary
   configuration variables in sacred are updated to include new values rather than
   overwriting the entire dictionary if the same configuration variable is specified
   multiple times. This means we can simulate the dynamic ingredients using this
   procedure as a workaround.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   examples.ingredients.sample_ingredient.generate_sample
   examples.ingredients.sample_ingredient.generate_admissible_actions



Attributes
~~~~~~~~~~

.. autoapisummary::

   examples.ingredients.sample_ingredient.sample_ingredient


.. py:data:: sample_ingredient




.. py:function:: generate_sample(seed, env, sample_space, sample_policy, action_space, _log)

   Generate a sample based on the ingredient config.

   Generates a sample based on the ingredient configuration variables. For instance, if
   the `sample_space` key `"sample_scheme"` is `"random"`, then the initial conditions
   of the sample are chosen randomly. A similar pattern follows for the `action_space`.
   The `sample_policy` determines the type of policy applied to the system during
   sampling.

   :param seed: Unused.
   :param env: The dynamical system model.
   :param sample_space: The sample space configuration variable.
   :param sample_policy: The sample policy configuration variable.
   :param action_space: The action_space configuration variable.

   :returns: A sample of observations taken from the system evolution.


.. py:function:: generate_admissible_actions(seed, env, action_space, _log)

   Generate a collection of admissible control actions.
