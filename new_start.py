import retro
import cv2
import numpy as np
import tensorflow as tf



from tf_agents.agents.dqn import dqn_agent
from tf_agents.metrics import tf_metrics
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils
from tf_agents.networks import q_network
from tf_agents.drivers import dynamic_step_driver

from tf_agents.specs import array_spec

from gym import Env
from gym.spaces import MultiBinary, Box

import matplotlib.pyplot as plt


import warnings

from tf_agents.replay_buffers import tf_uniform_replay_buffer

warnings.filterwarnings("ignore")

def convertMBtoINT(action): #converting the action format to INT, understable forr model training
    res = 0
    for i in range(len(action)):
        res += action[i] * pow(2, i)
    return res


def convertINTtoMB(action, size = 12): #converting INT to action format, understendable for gym retro
    res = np.zeros(size)
    for i in range(size-1, -1, -1):
        res[i] = action % 2
        action = action//2
    return res

# implementing PyEnvironment for later wrapping
class StreetEnv(py_environment.PyEnvironment):

    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis",
                               use_restricted_actions=retro.Actions.FILTERED)
        self.iterations = GAME_ITERATIONS
        self.score = 0
        self.previous_frame = 0
        self._current_time_step = None

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(84, 84, 1), dtype=np.uint8, minimum=0, maximum=255, name='observation')

        self._episode_ended = False

    def render(self):
        self.game.render()

    def observation_spec(self):
        """Return observation_spec."""
        return self._observation_spec

    def action_spec(self):
        """Return action_spec."""
        return self._action_spec

    def _reset(self):
        """Return initial_time_step."""
        obs = self.game.reset()
        obs = self.preprocess(obs)

        self.previous_frame = obs
        self.score = 0
        self._episode_ended = False

        return ts.restart(obs)

    def _step(self, action):
        """Apply action and return new time_step."""
        if self._episode_ended:
            return self.reset()

        obs, reward, done, info = self.game.step(convertINTtoMB(action))
        if done:
            self._episode_ended = True
            self.game.close()

        obs = self.preprocess(obs)
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs
        reward = info['score'] - self.score
        self.score = info['score']

        if self._episode_ended:
            self.game.close()
            return ts.termination(frame_delta, reward)
        else:
            return ts.transition(frame_delta, reward=0.0, discount=1.0)

    @staticmethod
    def preprocess(observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        channels = np.reshape(resize, (84, 84, 1))
        return channels


