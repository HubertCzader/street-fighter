import warnings

from tf_agents.replay_buffers import tf_uniform_replay_buffer

warnings.filterwarnings("ignore")

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

GAME_ITERATIONS = 1

num_iterations = 10 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}


# helpers
replay_buffer = None
driver = None
tf_agent = None


def convertMBtoINT(action):
    res = 0
    for i in range(len(action)):
        res += action[i] * pow(2, i)
    return res


def convertINTtoMB(action, size = 12):
    res = np.zeros(size)
    for i in range(size-1, -1, -1):
        res[i] = action % 2
        action = action//2
    return res


def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


# needs to implement interface
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


def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))


def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)


@tf.function
def train_function():
    iterator = iter(dataset)

    #print(iterator.eval())

    tf_agent.train = common.function(tf_agent.train)
    tf_agent.train_step_counter.assign(0)

    final_time_step, policy_state = driver.run()

    for i in range(100):
        final_time_step, _ = driver.run(final_time_step, policy_state)

    episode_len = []
    step_len = []
    res_loss = []
    for i in range(num_iterations):
        print(i)
        final_time_step, _ = driver.run(final_time_step, policy_state)
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, tf_agent.collect_policy)

        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience=experience)

        step = tf_agent.train_step_counter

        #if step.eval() % log_interval == 0:
        #print('step = {0}: loss = {1}'.format(step, train_loss.loss))
        #    episode_len.append(train_metrics[3].result())
         #   step_len.append(step)

        res_loss.append(train_loss.loss)
        #driver.

        #rint('Average episode length: {}'.format(train_metrics[3].result()))

        # problem with eval. cant have two enviroments at one time
    #   if step % eval_interval == 0:
    #  avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    # print('step = {0}: Average Return = {1}'.format(step, avg_return))


    #env.close()
    print("Train end")
    return res_loss


def demo(eval_env, eval_py_env, policy, sess):
    #tf.compat.v1.enable_eager_execution()

    with sess.as_default():
        for _ in range(1):
            time_step = eval_env.reset()

            while not time_step.is_last().eval():
              #  print(time_step.is_last().eval())
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                #print(time_step)
                eval_py_env.render()
                #time.sleep(0.01)



if __name__ == "__main__":
    sess = tf.compat.v1.Session()
    #tf.compat.v1.enable_eager_execution()

    tf.compat.v1.enable_resource_variables()
    env = StreetEnv()

    # this line ensures, the env is correctly setup, for now commented
    # utils.validate_py_environment(env, episodes=5)

    train_env = tf_py_environment.TFPyEnvironment(env)

    fc_layer_params = (100,)

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v2.Variable(0)

    tf_agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        train_step_counter=train_step_counter)

    print("INITIALIZING==========================================")
    tf_agent.initialize()
    print("DONE==========================================")

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    print("Batch Size: {}".format(train_env.batch_size))

    replay_observer = [replay_buffer.add_batch]

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=0,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=1)

    init = tf.global_variables_initializer()

    sess.run(init)

    with sess.as_default():
        #with tf.device('/device:GPU:0'):
        loss_story = train_function()

        # trying to get learning data
       # env.game.close()
       # print(loss_story[0].eval())
           # print(step_len[0].eval())

    #env.game.close()
    env.game.close()

    eval_py_env = StreetEnv()
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    demo(eval_env, eval_py_env, tf_agent.policy, sess)

