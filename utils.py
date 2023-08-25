from multiprocessing import Pipe, Process

import gym
import numpy as np


class VecEnv:
    def __init__(self, env_name, num_envs):
        self.num_envs = num_envs
        self.envs = [self._make_env(env_name) for _ in range(num_envs)]
        self.remotes, self.work_remotes = zip(*self.envs)

    def _make_env(self, env_name):
        remote, work_remote = Pipe()
        env = gym.make(env_name)
        process = Process(target=self._worker, args=(work_remote, env))
        process.start()
        return remote, process

    def _worker(self, remote, env):
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                remote.send((obs, reward, done, info))
            elif cmd == "reset":
                obs = env.reset()
                remote.send(obs)
            elif cmd == "close":
                remote.close()
                break

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return np.array(obs), np.array(rewards), np.array(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return np.array(obs)

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
