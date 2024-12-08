import gymnasium as gym
import os
import pickle
import time
import torch.nn as nn

from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from pyboy import PyBoy

from Environment import MarioPyBoyEnv, make_env
from Callbacks import ChartingCallback, ChartingCallbackMulti

if __name__ == '__main__':
    LOAD2 = "MarioWorld4NoTrainingScores4"
    LOAD = "MarioLandCrossNewScores5"

    if os.path.exists(LOAD):
        print("Old Scores Found, Continuing")
        with open(LOAD, 'rb') as file:
            SCORES = pickle.load(file)
    else:
        print(f"No File")
        exit()

    if os.path.exists(LOAD2):
        print("Old Scores Found, Continuing")
        with open(LOAD2, 'rb') as file:
            SCORES2 = pickle.load(file)
    else:
        print(f"No File")
        exit()


     #Printing Graphs
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt


    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    
    ravg = running_mean(SCORES,2500)
    x2 = range(0,len(ravg))

    ravg2 = running_mean(SCORES2,2500)
    z2 = range(0,len(ravg2))

    x = range(0,len(SCORES))
    z = range(0,len(SCORES2))
    plt.plot(x, SCORES, label="World 4 Pre-Trained Model")
    plt.plot(z, SCORES2, label="World 4 New Model")
    plt.plot(x2, ravg, label="Rolling Average World 4 Pre-Trained Model")
    plt.plot(z2, ravg2, label="Rolling Average World 4 New Model")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title(f"Super Mario Land Cross-Training")
    plt.legend()
    
    plt.savefig(f'{LOAD}{len(SCORES)}.png', format='png')
    plt.show()