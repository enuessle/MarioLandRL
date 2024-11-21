import gymnasium as gym
import os
import pickle
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from pyboy import PyBoy

from Environment import MarioPyBoyEnv, make_env
from Callbacks import ChartingCallback, ChartingCallbackMulti

if __name__ == '__main__':
    #Hyper Parameters
    MAX_EPISODES = 10000
    SCORES = []
    SAVE = "MarioLandFullActionSpaceSave"
    SCORE_SAVE = f'{SAVE}Scores.pkl'

    LEARNING_RATE = 3e-4
    N_STEPS = 2048                       # Number of steps to run in the environment per update


    if os.path.exists(SCORE_SAVE):
        print("Old Scores Found, Continuing")
        with open(SCORE_SAVE, 'rb') as file:
            SCORES = pickle.load(file)

    supermarioland_rom = "SuperMarioLand_rom.gb"

    # Creates a Vector Environment for Multiprocessing
    def createVectorEnv(rom:str, numEnvironments:int, debug:bool = False):
        # Create a list of environment factories
        env_fns = [make_env(rom, debug) for _ in range(numEnvironments)]

        # Create the vectorized environment
        vec_env = SubprocVecEnv(env_fns)
        return vec_env




    num_envs = os.cpu_count() - 1
    vec_env = createVectorEnv(supermarioland_rom,num_envs, debug=False)

    # Stops training when the model reaches the maximum number of episodes
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=MAX_EPISODES)
    charting_rewards = ChartingCallbackMulti(SCORES, verbose=1)

    # Check if the model file exists before loading
    if os.path.exists(SAVE + ".zip"):
        model = PPO.load(SAVE, env=vec_env)
        print("Model loaded successfully.")
    else:
        print(f"No saved model Called {SAVE} Found. Starting with a new model.")
        # Create and train a new model if no saved model exists
        model = PPO("MlpPolicy", vec_env, verbose=0, learning_rate=LEARNING_RATE, n_steps=N_STEPS, device="cpu")
        # Train your model or perform other actions

    #print(f"Policy: {model.policy}")
    print(f"Device: {model.policy.device}")
    print("Starting Training:")

    start_time = time.time()
    model.learn(total_timesteps=1e10, callback=[callback_max_episodes, charting_rewards])
    elapsed_time = time.time() - start_time
    print(f"Learning {MAX_EPISODES} Episodes Done in {elapsed_time:.2f}s")


    # Save the agent
    model.save(SAVE)
    with open(SCORE_SAVE, 'wb') as file:
        pickle.dump(SCORES, file)

    #Printing Graphs
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    x = range(0,len(SCORES))
    plt.plot(x, SCORES, label="Scores per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title(f"Super Mario Land with PPO")
    plt.legend()
    
    plt.savefig(f'{SAVE}{len(SCORES)}.png', format='png')
    plt.show()
