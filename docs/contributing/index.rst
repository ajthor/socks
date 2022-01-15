************
Contributing
************

.. toctree::
    :hidden:

    new_examples
    new_benchmarks


Notice a bug? Have a feature idea? Interested in contributing an algorithm?

There are many ways to contribute to SOCKS, and we welcome and encourage collaborators
to point out `issues <https://github.com/ajthor/socks/issues>`_, `post ideas
<https://github.com/ajthor/socks/discussions>`_, and (preferably) submit pull requests.

SOCKS is currently under active (alpha) development, meaning we are still in the early
stages of ironing out the way things work and developing the underlying framework of
SOCKS. We have a lot of great inspiration from existing libraries such as `OpenAI gym
<https://github.com/openai/gym>`_, `scikit-learn
<https://github.com/scikit-learn/scikit-learn>`_, `scipy
<https://github.com/scipy/scipy>`_, and more, but are still working on finding better
ways to interact with the algorithms in SOCKS.

Get Involved
============

1. Number one is to submit pull requests. We :octicon:`heart;1em;sd-text-danger` pull
   requests.
2. Submit feedback using the GitHub `issues
   <https://github.com/ajthor/socks/issues>`_ page, and tag your suggestions or feedback
   so that we know whether you're asking a question, posting about a bug, or would like
   to propose the next big idea.
3. Discussion. We'd love to hear how you are using SOCKS, or how you plan to use it in
   your own research or projects. Check out the `discussions
   <https://github.com/ajthor/socks/discussions>`_ page to get in touch.
4. Documentation. We have made an effort to fully document SOCKS, but know there is
   always room for improvement. If you notice the documentation is lacking, or even if
   you notice a typo, we encourage you to reach out and submit an issue or a PR to help
   fix it.

How to Pull Request :octicon:`git-pull-request`
-----------------------------------------------

Want to contribute code? Thank you!

Not sure how to submit a pull request (PR)? Read on.

.. important::

    Before you begin, make sure that the code you want to change or the issue you want
    to fix has not already been addressed by looking at the ``develop`` branch of the
    repo. ``develop`` contains all of the updated changes to the codebase that will
    eventually be turned into the next version of SOCKS.

Pull requests are the main way of contributing to open-source projects on GitHub. Below you will find a short tutorial on how to edit the code.

1. Fork the repository. On the `GitHub repo <https://github.com/ajthor/socks>`_, use
   the |fork| button near the top of the page. This will create a copy of the repo on
   your GitHub account.
2. Clone the fork to your system. An easy way to do this is using

   .. code-block:: shell

       git clone https://github.com/youraccount/socks
       cd socks

3. Make a new branch off of ``develop``. This will ensure that you are using the latest
   code.

   .. code-block:: shell

       git checkout develop
       git checkout -b your_branch_name

4. Make changes and commit. Make as many changes and commits as you like! We review all
   code changes when you submit the PR and squash the commits when it is merged.
5. Push the changes to your forked repo.
6. On GitHub, click the |pull-request| button (above the list of files).

.. |fork| raw:: html

    <span class="sd-sphinx-override sd-badge sd-outline-secondary sd-text-secondary"><svg aria-hidden="true" class="sd-octicon sd-octicon-git-fork" height="1.0em" version="1.1" viewbox="0 0 24 24" width="1.0em"><path d="M12 21a1.75 1.75 0 110-3.5 1.75 1.75 0 010 3.5zm-3.25-1.75a3.25 3.25 0 106.5 0 3.25 3.25 0 00-6.5 0zm-3-12.75a1.75 1.75 0 110-3.5 1.75 1.75 0 010 3.5zM2.5 4.75a3.25 3.25 0 106.5 0 3.25 3.25 0 00-6.5 0zM18.25 6.5a1.75 1.75 0 110-3.5 1.75 1.75 0 010 3.5zM15 4.75a3.25 3.25 0 106.5 0 3.25 3.25 0 00-6.5 0z" fill-rule="evenodd"></path><path d="M6.5 7.75v1A2.25 2.25 0 008.75 11h6.5a2.25 2.25 0 002.25-2.25v-1H19v1a3.75 3.75 0 01-3.75 3.75h-6.5A3.75 3.75 0 015 8.75v-1h1.5z" fill-rule="evenodd"></path><path d="M11.25 16.25v-5h1.5v5h-1.5z" fill-rule="evenodd"></path></svg> Fork</span>

.. |pull-request| raw:: html

    <span class="sd-sphinx-override sd-badge sd-outline-secondary sd-text-secondary"><svg aria-hidden="true" class="sd-octicon sd-octicon-git-pull-request" height="1.0em" version="1.1" viewbox="0 0 16 16" width="1.0em"><path d="M7.177 3.073L9.573.677A.25.25 0 0110 .854v4.792a.25.25 0 01-.427.177L7.177 3.427a.25.25 0 010-.354zM3.75 2.5a.75.75 0 100 1.5.75.75 0 000-1.5zm-2.25.75a2.25 2.25 0 113 2.122v5.256a2.251 2.251 0 11-1.5 0V5.372A2.25 2.25 0 011.5 3.25zM11 2.5h-1V4h1a1 1 0 011 1v5.628a2.251 2.251 0 101.5 0V5A2.5 2.5 0 0011 2.5zm1 10.25a.75.75 0 111.5 0 .75.75 0 01-1.5 0zM3.75 12a.75.75 0 100 1.5.75.75 0 000-1.5z" fill-rule="evenodd"></path></svg> Pull request</span>


.. attention::

    Make sure that you select the ``develop`` branch in the base repository to merge
    into.

Be sure to check out the guides on `GitHub <https://docs.github.com/en/pull-requests>`_
for more information.

Contributor Covenant
====================

We try our best to adhere to the |Contributor_Covenant|_, meaning, in short:

    We pledge to act and interact in ways that contribute to an open, welcoming,
    diverse, inclusive, and healthy community.

We aim to make SOCKS a positive environment for all.



However, while we encourage users to submit bug reports, questions, feature requests,
and PRs, we do not consider issues that are "homework" questions or "help solve my
specific problem" type issues--we simply don't have time. These issues will be marked as
:bdg-secondary:`off-topic` or :bdg-secondary:`wontfix`.

.. |Contributor_Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg
.. _Contributor_Covenant: https://www.contributor-covenant.org/version/2/1/code_of_conduct/