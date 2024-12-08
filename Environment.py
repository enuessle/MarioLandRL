import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
import random

# Environment Details
actions = [[''],['a'], ['b'], ['left'], ['right'], ['right', 'a'],['left', 'a'], ['right', 'b'],['left', 'b'], ['right', 'a', 'b'],['left', 'a', 'b'], ['up'], ['down']]
#actions = [[''], ['left'], ['right'], ['right', 'b'],['left', 'b'], ['right', 'a', 'b'],['left', 'a', 'b']]
matrix_shape = (16, 20)
game_area_observation_space = spaces.Box(low=0, high=255, shape=matrix_shape, dtype=np.uint32)

levels = [ (4,1)]

DEATH_PENALTY = 0

class MarioPyBoyEnv(gym.Env):

    def __init__(self, rom:str, fitness_threshold: int = 1000, debug=False, level = None):
        super().__init__()

        self.rom = rom
        if debug:
            pyboy = PyBoy(self.rom)
        else:
            pyboy = PyBoy(self.rom,window="null")  # Use headless mode for speed
            
        self.pyboy = pyboy
        self._fitness_threshold = fitness_threshold
        self._no_improvement_steps = 0  # Counter for steps without fitness improvement
        self._fitness=0
        self._previous_fitness=0
        self.debug = debug

        
        self.pyboy.set_emulation_speed(0)
        if debug:
            self.pyboy.set_emulation_speed(1)
        

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = game_area_observation_space

        self.level = level
        if self.level == None:
            level = random.choice(levels)
        self.pyboy.game_wrapper.set_world_level(level[0], level[1])  #Starting level
        self.pyboy.game_wrapper.start_game()
        self.pyboy.game_wrapper.game_area_mapping(self.pyboy.game_wrapper.mapping_minimal, 0)
        self.pyboy.game_wrapper.set_lives_left(0) #Start With One Life
        self._current_world = self.pyboy.game_wrapper.world

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Move the agent
        if action == 0:
            pass
        else:
            for a in actions[action]:
                self.pyboy.button(a)

        # Consider disabling renderer when not needed to improve speed:
        # self.pyboy.tick(1, False)
        if self.debug:
            self.pyboy.tick(1)
        else:
            self.pyboy.tick(1, False)

        done = self.pyboy.game_wrapper.game_over()

        # Level Changed (Level Complete)
        if self._current_world != self.pyboy.game_wrapper.world:
            done = True



        self._calculate_fitness()
        reward=self._fitness-self._previous_fitness
        if done:
            reward -= DEATH_PENALTY
        
        # Reward Moving to the Next World, Fix issue where new world drops score super large
        if self._current_world != self.pyboy.game_wrapper.world:
            self._current_world = self.pyboy.game_wrapper.world
            reward = 100
        if reward <= -1000:
            reward = 0
        # Check if fitness improved, if not, increment the no_fitness_improvement_steps counter
        if reward <= 0:
            self._no_improvement_steps += 1
        else:
            self._no_improvement_steps = 0  # Reset counter if fitness improved

        if self._no_improvement_steps > self._fitness_threshold:
            done = True
            if self.debug:
                print(f"Fitness hasn't improved for {self._fitness_threshold} steps. Ending episode.")
            self._no_improvement_steps = 0

        observation=self.pyboy.game_area()
        #observation = self.normalize_observation(observation)
        info = {"progress":self._fitness, 'level':self.pyboy.game_wrapper.world}
        truncated = False

        if done and self.debug:
            print(f"Done: fitness:{self._fitness}, progress:{self.pyboy.game_wrapper.level_progress}")

        return observation, reward, done, truncated, info

    def _calculate_fitness(self):
        self._previous_fitness=self._fitness

        # NOTE: Only some game wrappers will provide a score
        # If not, you'll have to investigate how to score the game yourself
        self._fitness=self.pyboy.game_wrapper.level_progress

    def reset(self, **kwargs):
        if self.debug:
            self.pyboy = PyBoy(self.rom)
            #self.pyboy.tick(render=True)
        else:
            self.pyboy = PyBoy(self.rom,window="null")  # Use headless mode for speed
        
        level = self.level
        if self.level == None:
            level = random.choice(levels)
        self.pyboy.game_wrapper.set_world_level(level[0], level[1])  #Starting level
        self.pyboy.game_wrapper.start_game()
        self.pyboy.game_wrapper.reset_game()
        self.pyboy.game_wrapper.game_area_mapping(self.pyboy.game_wrapper.mapping_minimal, 0)
        self.pyboy.game_wrapper.set_lives_left(0) #Start With One Life
        self._current_world = self.pyboy.game_wrapper.world
        self._fitness=0
        self._previous_fitness=0

        observation=self.pyboy.game_area()
        #observation = self.normalize_observation(observation)
        info = {}
        return observation, info
    
    def normalize_observation(self, observation):
        return observation / 255.0

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()



def make_env(rom: str, debug: bool = False):
    """
    Factory function to create a MarioPyBoyEnv instance.
    """
    def _init():
        from pyboy import PyBoy
        return MarioPyBoyEnv(rom, debug=debug)
    return _init

'''
pyboy = PyBoy(supermarioland_rom)
pyboy.set_emulation_speed(0)
assert pyboy.cartridge_title == "SUPER MARIOLAN"
mario = pyboy.game_wrapper
mario.game_area_mapping(mario.mapping_compressed, 0)
mario.start_game()
assert mario.score == 0
assert mario.lives_left == 2
assert mario.time_left == 400
assert mario.world == (1, 1)
last_time = mario.time_left
pyboy.tick() # To render screen after `.start_game`
pyboy.screen.image.save("SuperMarioLand1.png")
print(mario)
'''