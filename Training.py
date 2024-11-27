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
    #Hyper Parameters
    EPISODES_PER_SAVE = 5000
    SAVE_NAME = "MarioLandLongRunning"
    MAX_EPISODES = 10000
    SCORES = []
    #LOAD = "MarioLandRandomStartFourthIteration"
    #SAVE = "MarioLandRandomStartFifthIteration"
    #SCORE_LOAD = f'{LOAD}Scores.pkl'
    #SCORE_SAVE = f'{SAVE}Scores.pkl'

    LEARNING_RATE = 3e-5
    N_STEPS = 2048                       # Number of steps to run in the environment per update
    BATCH_SIZE = 512
    ENT_COEF = 7e-03
    GAMMA = 0.995


    # Try Using SAC or TD3

    '''
    PPO_HYPERPARAMS = {
    "policy": "MultiInputPolicy",
    "batch_size": 512,  # TODO: try 256
    "clip_range": 0.2,
    "ent_coef": 7e-03,
    "gae_lambda": 0.98,
    "gamma": 0.995,
    "learning_rate": 3e-05,
    "max_grad_norm": 1,
    "n_epochs": 5,
    "n_steps": 2048,
    "vf_coef": 0.5,
    "policy_kwargs": dict(
        activation_fn=nn.ReLU,
        features_extractor_class=MarioLandExtractor,
        features_extractor_kwargs=dict(
            # will be changed later
            device="auto",
        ),
        net_arch=dict(pi=[2048, 2048], vf=[2048, 2048]),
        normalize_images=False,
        share_features_extractor=True,
    ),
}'''

    # Define custom policy kwargs
    POLICY_KWARGS = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Directly pass a dictionary
    )

    '''
    if os.path.exists(SCORE_LOAD):
        print("Old Scores Found, Continuing")
        with open(SCORE_LOAD, 'rb') as file:
            SCORES = pickle.load(file)
    '''

    supermarioland_rom = "SuperMarioLand_rom.gb"


    # Creates a Vector Environment for Multiprocessing
    def createVectorEnv(rom:str, numEnvironments:int, debug:bool = False):
        # Create a list of environment factories
        env_fns = [make_env(rom, debug) for _ in range(numEnvironments)]

        # Create the vectorized environment
        vec_env = SubprocVecEnv(env_fns)
        return vec_env




    

    '''
    # Check if the model file exists before loading
    if os.path.exists(LOAD + ".zip"):
        model = PPO.load(LOAD, env=vec_env, policy_kwargs=POLICY_KWARGS, verbose=0, gamma = GAMMA,
                         learning_rate=LEARNING_RATE, n_steps=N_STEPS, batch_size=BATCH_SIZE, ent_coef=ENT_COEF, device="cpu")
        print("Model loaded successfully.")
    else:
        print(f"No saved model Called {LOAD} Found. Starting with a new model.")
        # Create and train a new model if no saved model exists
        model = PPO("MlpPolicy", vec_env, policy_kwargs=POLICY_KWARGS, verbose=0, gamma = GAMMA,
                    learning_rate=LEARNING_RATE, n_steps=N_STEPS, batch_size=BATCH_SIZE, ent_coef=ENT_COEF, device="cpu")
        # Train your model or perform other actions

        
    #print(f"Policy: {model.policy}")
    print(f"Device: {model.policy.device}")
    print("Starting Training:")

    start_time = time.time()
    model.learn(total_timesteps=1e10, callback=[callback_max_episodes, charting_rewards])
    elapsed_time = time.time() - start_time
    print(f"Learning {MAX_EPISODES} Episodes Done in {elapsed_time:.2f}s")


    # Save the agent
    model.save(f"{SAVE}")
    with open(SCORE_SAVE, 'wb') as file:
        pickle.dump(SCORES, file)
    '''
        


    num_envs = os.cpu_count() - 1
    vec_env = createVectorEnv(supermarioland_rom,num_envs, debug=False)

    # Training for EPISODES_PER_SAVE episodes, saving each time, then repeating
    
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=EPISODES_PER_SAVE)
    charting_rewards = ChartingCallbackMulti(SCORES, verbose=1)

    i = 0
    while(True):
        if ( i == 0 ):
            # New Model For First Iteration
            print(f"First Iteration, Making New Model")
            model = PPO("MlpPolicy", vec_env, policy_kwargs=POLICY_KWARGS, verbose=0, gamma = GAMMA,
                    learning_rate=LEARNING_RATE, n_steps=N_STEPS, batch_size=BATCH_SIZE, ent_coef=ENT_COEF, device="cpu")
        else:
            # Load Previous Iteration Model
            save = SAVE_NAME + f"{i}"
            model = PPO.load(save, env=vec_env, policy_kwargs=POLICY_KWARGS, verbose=0, gamma = GAMMA,
                         learning_rate=LEARNING_RATE, n_steps=N_STEPS, batch_size=BATCH_SIZE, ent_coef=ENT_COEF, device="cpu")
            print("Model loaded successfully.")

        # Train Model
        print(f"\n\n===============\nTraining Iteration {i}\n===============\n")
        start_time = time.time()
        model.learn(total_timesteps=1e10, callback=[callback_max_episodes, charting_rewards])
        elapsed_time = time.time() - start_time
        print(f"Learning {EPISODES_PER_SAVE} Episodes Done in {elapsed_time:.2f}s")

        # New Model Iteration Save
        i+=1

        # Save the agent
        save = SAVE_NAME + f"{i}"
        scoreSave =  SAVE_NAME + f"Scores{i}"
        model.save(f"{save}")
        with open(scoreSave, 'wb') as file:
            pickle.dump(SCORES, file)




    #Printing Graphs
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt


    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    
    ravg = running_mean(SCORES,2500)
    x2 = range(0,len(ravg))

    x = range(0,len(SCORES))
    plt.plot(x, SCORES, label="Scores per Episode")
    plt.plot(x2, ravg, label="Rolling Average")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title(f"Super Mario Land with PPO")
    plt.legend()
    
    plt.savefig(f'{SAVE}{len(SCORES)}.png', format='png')
    plt.show()
