import torch
import torch.nn as nn
from typing import Callable

class ObsNormalizer(nn.Module):
    """
    Normalizes observations using a running mean and variance.
    """
    def __init__(self, shape, eps=1e-8):
        super().__init__()
        self.count = 0
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.eps = eps

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = torch.clamp(new_var, min=self.eps)
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (self.var + self.eps).sqrt()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    """
    def func(progress: float) -> float:
        return progress * initial_value
    return func

class RolloutBufferMA:
    """
    Rollout buffer for multi-agent PPO.
    """
    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int, num_agents: int, sequence_length: int, device: torch.device):
        self.max_size = buffer_size
        self.n_agents = num_agents
        self.device = device
        self.obs = torch.zeros(buffer_size, num_agents, sequence_length, obs_dim, device=device)
        self.actions = torch.zeros(buffer_size, num_agents, sequence_length, act_dim, device=device)
        self.rewards = torch.zeros(buffer_size, num_agents, device=device)
        self.dones = torch.zeros(buffer_size, num_agents, device=device)
        self.values = torch.zeros(buffer_size, num_agents, device=device)
        self.logprobs = torch.zeros(buffer_size, num_agents, device=device)
        self.ptr = 0
        self.full = False

    def add(self, obs, action, reward, done, value, logprob):
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value
        self.logprobs[idx] = logprob.squeeze(1)
        self.ptr += 1
        if self.ptr >= self.max_size:
            self.full = True
            self.ptr = 0

    def compute_returns_adv(self, last_values: torch.Tensor, gamma: float, lam: float):
        T = self.max_size
        N = self.n_agents
        self.advantages = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)
        last_adv = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            non_terminal = 1.0 - self.dones[t]
            next_value = last_values if t == T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * non_terminal - self.values[t]
            last_adv = delta + gamma * lam * last_adv * non_terminal
            self.advantages[t] = last_adv
            self.returns[t] = last_adv + self.values[t]
        mean = self.advantages.mean(dim=0, keepdim=True)
        std = self.advantages.std(dim=0, keepdim=True) + 1e-8
        self.advantages = (self.advantages - mean) / std

    def get_batches(self, batch_size: int):
        idxs = torch.randperm(self.max_size, device=self.device)
        for start in range(0, self.max_size, batch_size):
            mb = idxs[start:start + batch_size]
            yield {
                'obs': self.obs[mb],
                'actions': self.actions[mb],
                'logp_old': self.logprobs[mb],
                'adv': self.advantages[mb],
                'returns': self.returns[mb],
                'values': self.values[mb],
            }

    def clear(self):
        self.ptr = 0
        self.full = False



class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        sequence_length: int,
        device: torch.device
    ):
        self.capacity = capacity
        self.n_agents = n_agents
        self.sequence_length = sequence_length
        self.device = device

        self.ptr = 0
        self.size = 0

        self.obs = torch.zeros(
            capacity, n_agents, sequence_length, obs_dim, device=device
        )
        self.next_obs = torch.zeros(
            capacity, n_agents, sequence_length, obs_dim, device=device
        )
        self.actions = torch.zeros(
            capacity, n_agents, sequence_length, act_dim, device=device
        )
        self.rewards = torch.zeros(
            capacity, n_agents, device=device
        )
        self.dones = torch.zeros(
            capacity, n_agents, device=device
        )

    def add(self, obs, action, reward, next_obs, done):
        """
        obs, next_obs: [N, L, D]
        action:        [N, A]
        reward, done:  [N]
        """
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.obs[idx],        # [B, N, L, D]
            self.actions[idx],    # [B, N, A]
            self.rewards[idx],    # [B, N]
            self.next_obs[idx],   # [B, N, L, D]
            self.dones[idx],      # [B, N]
        )

    def __len__(self):
        return self.size
    
def _build_actor_input(obs_window: torch.Tensor, actor_critic_type:str):
    """Return the list of per‑agent tensors fed to the actor."""
    if actor_critic_type in ('attention_based', 'Recurrent_based'):
        # full window
        return [obs_window[i].unsqueeze(0) for i in range(obs_window.size(0))]
    elif actor_critic_type == 'MLP_based':
        return [obs_window[i,-1].unsqueeze(0) for i in range(obs_window.size(0))]
    else:
        raise ValueError(actor_critic_type)