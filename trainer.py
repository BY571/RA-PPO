import random

import gym
import numpy as np
import torch

import wandb

from agent import PPOAgent
from buffer import PPOBuffer
from utils import VecEnv


class Trainer:
    def __init__(self, config):
        super().__init__()

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

        self.config = config
        self.max_ep_len = config.max_ep_len
        self.update_frequency = config.update_frequency
        self.num_episodes = config.num_episodes

        # TODO: make those vec envs
        self.train_env = VecEnv(config.env, self.config.num_envs)
        self.test_env = gym.make(config.env)

        observation_space = self.test_env.observation_space
        self.action_space = self.test_env.action_space

        self.agent = PPOAgent(observation_space, self.action_space, config)

        self.buffer = PPOBuffer(
            buffer_size=self.update_frequency,
            num_envs=self.config.num_envs,
            state_dim=observation_space.shape[0],
            action_space=self.action_space,
            device=config.device,
        )

        self.updating_steps = 0

    def eval(
        self,
    ):
        return_G = 0
        state = self.test_env.reset()
        while True:
            action = self.agent.get_action(state)
            state, reward, done, _ = self.test_env.step(action)
            return_G += reward
            if done:
                break

        return return_G

    def train(
        self,
    ):
        with wandb.init(
            project=self.config.project,
            name=self.config.run_name,
            config=self.config,
            mode=self.config.mode,
        ):
            max_interactions = self.config.max_interactions
            current_step = 0

            eval_requency = self.config.eval_frequency
            total_episodes = 0

            # init eval
            test_return = self.eval()
            wandb.log(
                {"episode": total_episodes, "test_return": test_return},
                step=current_step,
            )

            while current_step < max_interactions and total_episodes < self.num_episodes:
                total_reward = [0 for _ in range(self.config.num_envs)]
                state = self.train_env.reset()
                total_returns = []
                for t in range(self.max_ep_len):
                    action, log_prob = self.agent.get_action_log_prob(state)

                    # Collection step
                    next_state, reward, done, _ = self.train_env.step(
                        action.cpu().numpy()
                    )

                    self.buffer.add(state, action, log_prob, reward, next_state, done)

                    total_reward += reward
                    state = next_state
                    current_step += 1 * self.config.num_envs

                    if current_step % self.update_frequency == 0:
                        experiences = self.buffer.sample()
                        loss_info = self.agent.update(*experiences)
                        self.buffer.reset()
                        loss_info["episode"] = total_episodes
                        wandb.log(loss_info, step=current_step)

                    if current_step % eval_requency == 0:
                        test_return = self.eval()
                        wandb.log(
                            {"episode": total_episodes, "test_return": test_return},
                            step=current_step,
                        )

                    if done.any():
                        total_returns.append(total_reward.mean())
                        total_episodes += 1 * self.config.num_envs
                        train_log_info = {
                            "episode": total_episodes,
                            "train_return": total_reward.mean(),
                        }
                        wandb.log(train_log_info, step=current_step)

                        print(
                            "Episode: {} -- train_return: {}".format(
                                total_episodes, total_reward.mean()
                            )
                        )
                        break

            print("Training finished!")
            self.train_env.close()
