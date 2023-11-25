import retro
import warnings
import time

warnings.filterwarnings("ignore")

GAME_ITERATIONS = 1


class StreetFighter:

    def __init__(self, iterations):
        self.env = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis")
        self.iterations = iterations

    def run(self):
        self.env.reset()
        done = False
        for game in range(self.iterations):
            while not done:
                if done:
                    self.env.reset()
                self.env.render()
                obs, reward, done, info = self.env.step(self.env.action_space.sample())
                time.sleep(0.01)
        self.env.close()


if __name__ == "__main__":
    Game = StreetFighter(GAME_ITERATIONS)
    Game.run()
