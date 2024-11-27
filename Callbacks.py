from stable_baselines3.common.callbacks import BaseCallback
import time

from Environment import DEATH_PENALTY

class ChartingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, scores, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards_list = scores
        self.episode_reward = 0
        self.episode = 0
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        # Accumulate rewards for the current episode
        self.episode_reward += self.locals["rewards"][0]

        # Check if the episode is done
        if self.locals["dones"][0]:
            # Append the total reward for this episode to the external list
            self.episode_rewards_list.append(self.episode_reward)
            #Print
            print(f"Episode {self.episode}:      Score {self.episode_reward}")
            # Reset episode reward for the next episode
            self.episode_reward = 0
            self.episode+=1

        return True
    

class ChartingCallbackMulti(BaseCallback):
    """
    A custom callback that tracks the rewards for each environment and prints:
    - Episode rewards for each environment.
    - Average reward across all environments.
    - Real-life time between each average score calculation.

    :param scores: A list to store the average rewards per episode index.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages.
    """
    def __init__(self, scores, verbose: int = 0):
        super().__init__(verbose)
        self.scores = scores  # External list to store average scores per "batch" of episodes
        self.environmentScores = [] # Scores Seperated by Environment
        self.environmentEpisodeNum = [] # Keep Track of the Episode of each Environment
        self.start_time = None  # Track the start time for elapsed time
        self.last_time = None  # Last time an average score was calculated
        self.count = 0
        
    def _on_training_start(self) -> None:
        """
        Initialize tracking for all environments and time when training starts.
        """
        num_envs = self.training_env.num_envs  # Number of parallel environments
        self.environmentScores = [[0.0] for _ in range(num_envs)] # Scores Seperated by Environment
        self.environmentEpisodeNum = [0] * num_envs # Keep Track of the Episode of each Environment

        #Timing Variables
        self.start_time = time.time()
        self.last_time = self.start_time

    def _on_step(self) -> bool:
        """
        Called after each environment step to update rewards and track episode completions.
        """
        rewards = self.locals["rewards"]  # Rewards for each environment
        dones = self.locals["dones"]  # Done flags for each environment
        infos = self.locals["infos"]

        for i, reward in enumerate(rewards):
            #Episode Num
            episodeNum = self.environmentEpisodeNum[i]
            # Accumulate reward for the current episode
            self.environmentScores[i][episodeNum] += reward

            if dones[i]:  # If this environment's episode is done
                # Log the environment's reward
                if self.verbose > 1:
                    print(f"Environment {i} - Episode {episodeNum+1} - Level {infos[i]['level']} - Reward: {self.environmentScores[i][episodeNum]+ DEATH_PENALTY:.2f}")
                    # print(f"Info: {infos[i]['progress']}")

                if self.verbose > 0:
                    if self.count >= self.training_env.num_envs:
                        print(f"Environment {i} - Episode {episodeNum+1} - Level {infos[i]['level']} - Reward: {self.environmentScores[i][episodeNum]+ DEATH_PENALTY:.2f}")
                        # print(f"Info: {infos[i]['progress']}")
                        self.count = 0
                    self.count+=1


                # Update Episode Number for that Environment
                self.scores.append(self.environmentScores[i][episodeNum] + DEATH_PENALTY)
                self.environmentEpisodeNum[i] += 1
                self.environmentScores[i].append(0.0)

                # If This episode now has all scores for all environments, calculate the average
                '''
                valid = all(env_episode_num > episodeNum for env_episode_num in self.environmentEpisodeNum)
                if valid:
                    average_reward = sum(self.environmentScores[i][episodeNum] for i in range(len(self.environmentScores))) / self.training_env.num_envs
                    if self.verbose > 2:
                        print(f"{sum(self.environmentScores[i][episodeNum] for i in range(len(self.environmentScores)))}, {self.training_env.num_envs}")
                    self.scores.append(average_reward)

                    # Calculate real-life time elapsed since the last average
                    current_time = time.time()
                    elapsed_time = current_time - self.last_time
                    self.last_time = current_time

                    # Print average and elapsed time
                    if self.verbose > 0:
                        print(f"Average Reward For Episode {len(self.scores)}: {average_reward:.2f} | Time Elapsed: {elapsed_time:.2f}s")
                '''

        return True