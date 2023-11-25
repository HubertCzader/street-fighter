import retro
import warnings
import time

warnings.filterwarnings("ignore")

GAME_ITERATIONS = 1


def run_game(iterations: int = GAME_ITERATIONS):
    env = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis")
    env.reset()
    done = False
    for game in range(iterations):
        while not done:
            if done:
                env.reset()
            env.render()
            obs, reward, done, info = env.step(env.action_space.sample())
            time.sleep(0.01)
    env.close()


def preprocessing():
    pass


if __name__ == "__main__":
    run_game()
