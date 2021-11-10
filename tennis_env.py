"""
wrapper to give a unity environment the same interface as aigym to
facilitate running the same agents

this multiagent environment has an observation space of 24 floats, which
is the result of stacking 3 frames of 8 floats each, representing the
position/velocity of the ball/racket.

the action space is 2 floats per player corresponding to movement toward
(or away from) the net, and jumping.
"""
import numpy as np
import gym
from unityagents import UnityEnvironment


class Environment:
    action_space = gym.spaces.Box(-1, 1, (2,))
    observation_space = gym.spaces.Box(-np.inf, np.inf, (24,))

    def __init__(self, env_id=0, filename='Tennis_Linux_NoVis/Tennis.x86_64'):
        self.env = UnityEnvironment(file_name=filename, no_graphics=True,
                                    worker_id=env_id)
        self.brain_name = self.env.brain_names[0]

    def reset(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        return env_info.vector_observations

    def step(self, actions):
        """ take an action from the agent and update the environment

        :param actions: an ndarray of shape (2,2).  each player's actions
            are 2 floats each.
        :returns: a tuple of arrays: next_state (24 floats),
            reward (float), done (int), None
        """
        env_info = self.env.step(actions)[self.brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done
        return next_state, reward, done, None

    def close(self):
        """ shut down unity """
        self.env.close()
