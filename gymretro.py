import retro
import warnings
import time
import cv2
import numpy as np

from gym import Env
from gym.spaces import MultiBinary, Box

warnings.filterwarnings("ignore")

GAME_ITERATIONS = 1


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
                # time.sleep(0.01)
        self.close()


if __name__ == "__main__":
    Game = StreetFighter()
    Game.run()
