import torch
from gym.spaces import Box, Discrete


class PPOBuffer:
    """Fixed-size buffer to store experience tuples from parallel environments."""

    def __init__(self, buffer_size, num_envs, state_dim, action_space, device):
        """Initialize a PPOBuffer object."""

        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self.idx = 0
        if isinstance(action_space, Box):
            self.action_dtype = torch.float32
            self.action_dim = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            self.action_dtype = torch.int64
            self.action_dim = 1

        # Preallocate memory
        self.states = torch.zeros(
            (buffer_size, num_envs, state_dim), dtype=torch.float32
        ).to(device)
        self.actions = torch.zeros(
            (buffer_size, num_envs, self.action_dim), dtype=self.action_dtype
        ).to(device)
        self.log_probs = torch.zeros((buffer_size, num_envs), dtype=torch.float32).to(
            device
        )
        self.rewards = torch.zeros((buffer_size, num_envs), dtype=torch.float32).to(
            device
        )
        self.next_states = torch.zeros(
            (buffer_size, num_envs, state_dim), dtype=torch.float32
        ).to(device)
        self.dones = torch.zeros((buffer_size, num_envs), dtype=torch.float32).to(
            device
        )

    def add(self, states, actions, log_probs, rewards, next_states, dones):
        """Add a new batch of experiences to memory."""

        self.states[self.idx] = torch.tensor(states, dtype=torch.float32).to(
            self.device
        )
        self.actions[self.idx] = torch.tensor(actions, dtype=self.action_dtype).to(
            self.device
        )[:, None]
        self.log_probs[self.idx] = torch.tensor(log_probs, dtype=torch.float32).to(
            self.device
        )
        self.rewards[self.idx] = torch.tensor(rewards, dtype=torch.float32).to(
            self.device
        )
        self.next_states[self.idx] = torch.tensor(next_states, dtype=torch.float32).to(
            self.device
        )
        self.dones[self.idx] = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Move to the next position in the buffer
        self.idx = (self.idx + 1) % self.buffer_size

    def sample(self):
        """Return all stored experiences from memory up to current index."""
        return (
            self.states,  # [:self.idx],
            self.actions,  # [:self.idx],
            self.log_probs,  # [:self.idx],
            self.rewards,  # [:self.idx],
            self.next_states,  # [:self.idx],
            self.dones,
        )  # [:self.idx])

    def reset(self):
        """Reset the buffer by setting the index to zero."""
        self.idx = 0

    def __len__(self):
        """Return the current size of internal memory."""
        return self.idx
