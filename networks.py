import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class BaseActor(nn.Module):
    def __init__(self, observation_space, hidden_size=32):
        super(BaseActor, self).__init__()
        self.state_size = observation_space.shape[0]
        self.fc1 = nn.Linear(self.state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return x


class DiscreteActor(BaseActor):
    def __init__(self, observation_space, action_space, hidden_size=32):
        super(DiscreteActor, self).__init__(observation_space, hidden_size)
        self.action_size = action_space.n
        self.fc3 = nn.Linear(hidden_size, self.action_size)

    def forward(self, state):
        x = super().forward(state)
        x = self.fc3(x)
        return Categorical(logits=x)
    
    def evaluate_actions(self, state, action):
        dist = self.forward(state)
        log_prob = dist.log_prob(action)
        return log_prob, dist.entropy()

    def get_eval_action(self, state):
        dist = self.forward(state)
        return dist.probs.argmax(dim=-1)

    def get_action(self, state):
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class ContinuousActor(BaseActor):
    def __init__(self, observation_space, action_space, hidden_size=32):
        super(ContinuousActor, self).__init__(observation_space, hidden_size)
        self.action_size = action_space.shape[0]
        self.fc3 = nn.Linear(hidden_size, 2 * self.action_size)

    def forward(self, state):
        x = super().forward(state)
        x = self.fc3(x)
        mu, log_std = x.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        return Normal(mu, std)
    
    def evaluate_actions(self, state, action):
        dist = self.forward(state)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return log_prob, dist.entropy()

    def get_eval_action(self, state):
        dist = self.forward(state)
        return dist.mean

    def get_action(self, state):
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob.sum(dim=-1)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, observation_space, hidden_size=32):
        """Initialize parameters and build model.
        """
        super(Critic, self).__init__()
        state_size = observation_space.shape[0]
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        """Build a critic (value) network that maps states to Values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x), None
    
class DCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=32, n_cosines=64,):
        super(DCritic, self).__init__()
        self.state_size = observation_space.shape[0]
        self.action_size = action_space.n
        
        pis = torch.FloatTensor([np.pi*i for i in range(1, n_cosines+1)]).view(1, 1, n_cosines) # Starting from 0 as in the paper
        self.register_buffer('pis', pis)
        
        # Cosine embedding
        self.cosine_embedding_layer = nn.Linear(n_cosines, hidden_size)
        self.n_cos = n_cosines

        self.state_embedding_layer = nn.Linear(self.state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, state, n_taus=64, lb=0, ub=1):
        bs = state.shape[0]
        cos, taus = self.calc_cos(bs, n_taus, lb=lb, ub=ub)
        cos_embd = self.cosine_embedding_layer(cos)
        state_embd = self.state_embedding_layer(state).unsqueeze(1)
        # state_emb = shape (batch, 1, embed_dim)   
        comb_embd = (state_embd * cos_embd).reshape(bs*n_taus, -1)
        x = F.relu(self.fc2(comb_embd))
        return self.fc3(x).reshape(bs, n_taus, -1), taus.squeeze(-1)


    def calc_cos(self, batch_size, n_taus, lb=0, ub=1):
            """
            Calculating the cosinus values depending on the number of tau samples
            """
            taus = (torch.FloatTensor(batch_size, n_taus).uniform_(lb, 1) * ub) .unsqueeze(-1).to(self.pis.device)#(batch_size, n_tau, 1)  .to(self.device)
            cos = torch.cos(taus*self.pis)

            assert cos.shape == (batch_size, n_taus, self.n_cos), "cos shape is incorrect"
            return cos, taus
