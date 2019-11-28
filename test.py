# Simple env test.
import json
import select
import time
import logging
import os

import aicrowd_helper
import gym
import minerl

import coloredlogs
coloredlogs.install(logging.DEBUG)

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamond-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))

def main():
    """
    This function will be called for training phase.
    """
    # Sample code for illustration, add your code below to run in test phase.
    # Load trained model from train/ directory
    env = gym.make(MINERL_GYM_ENV)

    actions = [env.action_space.sample() for _ in range(10)]
    xposes = []
    for _ in range(MINERL_MAX_EVALUATION_EPISODES):
        obs = env.reset()
        done = False
        netr = 0
        while not done:
            random_act = env.action_space.noop()
            random_act['camera'] = [0, 0.3]
            random_act['back'] = 0
            random_act['forward'] = 1
            random_act['jump'] = 1
            random_act['attack'] = 1
            random_act['craft'] = 1
            random_act['equip'] = 1
            random_act['left'] = 0
            random_act['nearbyCraft'] = 0
            random_act['nearbySmelt'] = 0
            random_act['place'] = 0
            random_act['right'] = 0
            random_act['sneak'] = 0
            random_act['sprint'] = 0

            print (random_act)
            print (f'random_act: {random_act}')
            obs, reward, done, info = env.step(random_act)

            print (f'obs: {obs}')
            print (f'reward: {reward}')
            print (f'done: {done}')
            print (f'info: {info}')
            break
            netr += reward

        break

    env.close()

if __name__ == "__main__":
    main()
