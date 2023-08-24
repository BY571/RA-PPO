import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import Box, Discrete

from networks import DiscreteActor, ContinuousActor, Critic, DCritic


class PPOAgent():
    def __init__(self, observation_space, action_space, config):
        self.device = config.device
        self.gamma = config.gamma
        self.lam = config.lam
        self.clip_epsilon = config.clip_epsilon
        self.use_entropy = config.use_entropy
        self.entropy_coef = config.entropy_coef
        self.minibatch_size = config.minibatch_size
        self.num_updates = config.num_updates
        self.eta = config.eta
        self.use_meta_eta = config.use_meta_eta
        self.iqr_threshold = config.iqr_threshold
        self.num_envs = config.num_envs

        # Actor and Critic networks
        if isinstance(action_space, Box):
            self.actor = ContinuousActor(observation_space, action_space, config.hidden_size).to(self.device)
        elif isinstance(action_space, Discrete):
            self.actor = DiscreteActor(observation_space, action_space, config.hidden_size).to(self.device)
        else:
            raise NotImplementedError("Only Box and Discrete action spaces are supported")

        self.critic = DCritic(observation_space, action_space, config.hidden_size).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr) # eps=1e-3
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr) #  eps=1e-3
        self.vmap_gae = torch.vmap(self.compute_gae, in_dims=1, out_dims=1)

    @torch.no_grad()
    def get_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        return self.actor.get_eval_action(state).cpu().numpy()
    
    @torch.no_grad()
    def get_action_log_prob(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        return self.actor.get_action(state)

    @staticmethod
    def calculate_huber_loss(td_errors, k=1.0):
        loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
        return loss
    
    @staticmethod
    def calculate_quantile_huber_loss(huber_loss, td_errors, taus, kappa=1.0):
        return torch.abs(
        taus[..., None] - (td_errors.detach() < 0).float()
        ) * huber_loss / kappa

    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation (GAE)"""
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards).to(self.device)
        advantage = 0
        for t in reversed(range(len(rewards))):
            advantage = deltas[t, :] + self.gamma * self.lam * advantage * (1 - dones[t, :])
            advantages[t] = advantage
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return advantages

    def update(self, states, actions, log_probs_old, rewards, next_states, dones):
        """Update the policy and value parameters using PPO update rule."""

        eta = self.eta
         # meta adaptation
        with torch.no_grad():
            quantiles, _ = self.critic(states.flatten(0,1), n_taus=64, lb=0, ub=1)
            sorted_q = torch.sort(quantiles, dim=1)[0]
            # 0.75 quantile
            q_75 = sorted_q[:, int(0.75 * 64)]
            # 0.25 quantile
            q_25 = sorted_q[:, int(0.25 * 64)]
            iqr = q_75 - q_25
            if self.use_meta_eta:
                ones = torch.ones_like(iqr)
                adapted_meta_eta = torch.where(iqr > self.iqr_threshold, self.eta, ones)
                eta = adapted_meta_eta

        # Compute values for states and next_states
        with torch.no_grad():
            quantiles, _ = self.critic(states.flatten(0,1), n_taus=64, lb=0, ub=eta)
            quantiles = quantiles.reshape(-1, self.num_envs, 64, 1)
            next_quantiles, _ = self.critic(next_states.flatten(0,1), n_taus=64, lb=0, ub=eta)
            next_quantiles = next_quantiles.reshape(-1, self.num_envs, 64, 1)
        rewards = rewards[..., None]
        dones = dones[..., None]
        log_probs_old = log_probs_old[..., None]

        # Compute advantages using GAE
        quant_advantage = self.vmap_gae(rewards[..., None].expand(-1, self.num_envs, 64, 1), quantiles, next_quantiles, dones[..., None].expand(-1, self.num_envs, 64, 1))
        # advantages = self.vmap_gae(rewards, quantiles.mean(-2), next_quantiles.mean(-2), dones)
        target_value = quant_advantage + quantiles
        
        # stack all the data
        states = states.reshape(-1, states.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        log_probs_old = log_probs_old.reshape(-1, log_probs_old.shape[-1])
        advantages = quant_advantage.mean(-2).reshape(-1, quant_advantage.shape[-1])
        target_value = target_value.reshape(-1, 64, target_value.shape[-1])

        # PPO update loop
        for _ in range(self.num_updates):
            sample_indices = torch.randint(low=0, high=states.shape[0], size=(self.minibatch_size,))
            loss_info = self._update(states[sample_indices], actions[sample_indices], log_probs_old[sample_indices], advantages[sample_indices], target_value[sample_indices])

        loss_info["eta"] = adapted_meta_eta.mean().item() if self.use_meta_eta else eta
        loss_info["iqr"] = iqr.mean().item()
        return loss_info

    def _update(self, states, actions, log_probs_old, advantages, target_value):
        log_probs, entropy = self.actor.evaluate_actions(states, actions.squeeze())
        quantiles, taus = self.critic(states)

        ratio = (log_probs[:, None] - log_probs_old).exp()
        obj = ratio * advantages
        obj_clipped = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(obj, obj_clipped).mean() 
        
        if self.use_entropy:
            policy_loss -= 0.01 * entropy.mean()
        
        td_errors = target_value.transpose(1,2) - quantiles
        element_wise_huber_loss = self.calculate_huber_loss(td_errors) # (batch_size, 64, 64)

        quantile_huber_loss = self.calculate_quantile_huber_loss(element_wise_huber_loss, td_errors, taus) # (batch_size, 64, 64)
        value_loss = quantile_huber_loss.sum(dim=1).mean(dim=1, keepdim=True)
        # could now multiply with weights from per
        value_loss = value_loss.mean()
                    
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        # TODO add lr schedule

        return {"total_loss": policy_loss.item() + value_loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": entropy.mean().item()}

