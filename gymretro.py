import retro
import warnings
import time
from operator import itemgetter


import numpy as np


warnings.filterwarnings("ignore")

GAME_ITERATIONS = 1


class StreetFighter:

    def __init__(self, iterations):
        self.env = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis", obs_type=retro.Observations.RAM,
                              use_restricted_actions=retro.Actions.DISCRETE)
        self.iterations = iterations

    def run(self):
        #RAM size: 65536
        #ACTIONs can be discrete - restricted_actions - and filtered not allowed actions, maybe further filtering? Only buttons?

        l_obs = np.zeros(65536)
        print(l_obs)

        print(self.env.action_space)

        self.env.reset()
        done = False

        often_changes = dict()

        for game in range(self.iterations):
            while not done:
                if done:
                    self.env.reset()
                self.env.render()
                obs, reward, done, info = self.env.step(self.env.action_space.sample())

                delta_obs = obs - l_obs

                i_list = []

                for i in range(len(delta_obs)):
                    i_list.append((delta_obs[i], l_obs[i], obs[i], i))

                changes = [(x[1], x[2], x[3]) for x in i_list if x[0] != 0]
               # print(changes)


                for change in changes:
                    if change[2] in often_changes.keys():
                        often_changes[change[2]] += 1
                    else:
                        often_changes[change[2]] = 1
                print(sorted(often_changes.items(), key=itemgetter(1), reverse=True))
                print("==========================================")


                '''
                hi_list = []
                for i in range(len(obs)):
                    hi_list.append((obs[i], i))
                possible_health = [x[1] for x in hi_list if x[0] == info['health']]

                for index in possible_health:
                    if index in health_pool.keys():
                        health_pool[index] += 1
                    else:
                        health_pool[index] = 1

                print(possible_health)
                print(sorted(health_pool.items(), key=itemgetter(1), reverse=True))

                #(32834, 1339), (32832, 1334), (33000, 1207) - possible health indexes
                
                print("==========================================")
                print(info)
                l_obs = obs
                '''

                #time.sleep(0.01)
        self.env.close()


if __name__ == "__main__":
    Game = StreetFighter(GAME_ITERATIONS)
    Game.run()
