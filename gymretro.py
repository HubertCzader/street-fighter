import os.path
import warnings

warnings.filterwarnings("ignore")

import retro
import time
import cv2
import optuna
import numpy as np
import os
import tensorflow as tf

from gym import Env
from gym.spaces import MultiBinary, Box
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack

GAME_ITERATIONS = 1

# for hyperparameter optimization
LOG_DIR = "./logs/"
OPT_DIR = "./opt/"


# Function to return test hyperparameters
def optimize_ppo(trial):
    return{
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        #'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        #'gae_lambda': trial.suggest_int('gae_lambda', 0.8, 0.99)
    }

# training
def optimize_agent(trial):
    try:
        model_params = optimize_ppo(trial)

        #enviroment
        env = StreetFighter()
        env = Monitor(env, LOG_DIR)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4)

        #algorithm
        model = PPO2('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)
        model.learn(total_timesteps=300) # for training needs much more, like 100k
        #eval
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        env.close()

        SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_bes_model'.format(trial.number))
        model.save(SAVE_PATH)

        return mean_reward

    except Exception as e:
        print(e)
        return -1000

class StreetFighter(Env):

    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis",
                               use_restricted_actions=retro.Actions.FILTERED)
        self.iterations = GAME_ITERATIONS
        self.score = 0
        self.previous_frame = 0

    def reset(self):
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.score = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs
        reward = info['score'] - self.score
        self.score = info['score']
        return frame_delta, reward, done, info

    @staticmethod
    def preprocess(observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        channels = np.reshape(resize, (84, 84, 1))
        return channels

    def render(self, *args, **kwargs):
        self.game.render()

    def close(self):
        self.game.close()

    def run(self):
        self.game.reset()
        done = False
        for game in range(self.iterations):
            while not done:
                if done:
                    self.reset()
                self.render()
                obs, reward, done, info = self.step(self.action_space.sample())
                if reward > 0:
                    print(reward)
                time.sleep(0.01)
        self.close()


if __name__ == "__main__":
    #Game = StreetFighter()
    #Game.run()
    study = optuna.create_study(direction='maximize'))
    study.optimize(optimize_agent, n_trials=10, n_jobs=1)
