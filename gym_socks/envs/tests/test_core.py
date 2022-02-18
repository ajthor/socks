import unittest
from unittest.mock import patch

import gym

from gym_socks.envs import NDIntegratorEnv
from gym_socks.envs.core import BaseWrapper, pre_hook_wrapper
from gym_socks.envs.core import post_hook_wrapper

import numpy as np


def custom_pre_hook():
    # print("PRE")
    pass


def custom_post_hook():
    # print("POST")
    pass


class DummyWrapper(NDIntegratorEnv):
    @pre_hook_wrapper(custom_pre_hook)
    @post_hook_wrapper(custom_post_hook)
    def reset(self):
        return


class TestWrapper(unittest.TestCase):
    # @patch.object(HookWrapper, "pre_hook", new=custom_pre_hook)
    # @patch.object(HookWrapper, "post_hook", new=custom_post_hook)
    def test_pre_post_hook(cls):

        # mock_pre_hook.new_callable = custom_pre_hook
        # mock_post_hook.new_callable = custom_post_hook

        # wrapped = HookWrapper(NDIntegratorEnv())
        wrapped = DummyWrapper()

        wrapped.reset()

        pass

    def test_wrapper_class(cls):

        wrapped = BaseWrapper(NDIntegratorEnv(2))

        # print(wrapped.reset())
