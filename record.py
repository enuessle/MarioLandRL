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
import keyboard


# Load the model
MODEL = "MarioLandLongRunning10.zip"  # Replace with the actual model path
model = PPO.load(MODEL, device="cpu")
ROM = "SuperMarioLand_rom.gb"


levels = [ (1,1), (1,2), (1,3), (2,1), (2,2), (3,1), (3,2), (3,3), (4,1), (4,2)]
# Create and wrap the Mario environment
# Create and wrap the Mario environment in DummyVecEnv

for l in levels:
    def make_env():
        return MarioPyBoyEnv(ROM, debug=True, level=l)

    env = DummyVecEnv([make_env])
    obs = env.reset()

    # Run the model in the environment
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)  # Adjusted unpacking
        done = done[0]  # VecEnv returns an array of dones

    # Clean up
    env.close()