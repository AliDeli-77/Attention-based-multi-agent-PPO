import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from abc import ABC, abstractmethod

from src.utils import RolloutBufferMA, ReplayBuffer
from src.utils import _build_actor_input



# =========================
# Base Algorithm Interface
# =========================

class BaseAlgorithm(ABC):
    def __init__(self, actor, critic, cfg, device):
        self.actor = actor
        self.critic = critic
        self.cfg = cfg
        self.device = device

    @abstractmethod
    def select_action(self, obs, deterministic=False):
        pass

    @abstractmethod
    def store_transition(self, *args):
        pass

    @abstractmethod
    def update(self):
        pass


# =========================
# PPO (Multi-Agent)
# =========================

class MultiAgentPPO(BaseAlgorithm):
    def __init__(self, actor, critic, cfg, device):
        super().__init__(actor, critic, cfg, device)

        self.buffer = RolloutBufferMA(
            buffer_size=cfg.rollout_steps,
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            num_agents=cfg.num_agents,
            sequence_length=cfg.sequence_length,
            device=device
        )

        self.actor_optim = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = optim.Adam(critic.parameters(), lr=cfg.critic_lr)



    def select_action(self, obs_window, deterministic=False):
        obs_list = _build_actor_input(obs_window, self.cfg.actor_critic_type)

        with torch.no_grad():
            if self.cfg.actor_critic_type == "Recurrent_based":
                means, stds, self.hx_actor = self.actor(obs_list, self.hx_actor)
            else:
                means, stds = self.actor(obs_list)

        actions, logps = [], []
        for i in range(self.cfg.num_agents):
            dist = torch.distributions.Normal(
                means[i], torch.clamp(stds[i], 1e-3, 1.0)
            )
            a = dist.sample()
            lp = dist.log_prob(a).sum()
            actions.append(a.squeeze(0))
            logps.append(lp.unsqueeze(0))

        return torch.stack(actions), torch.stack(logps)


    def store_transition(self, obs_window, action, logp, reward, value, done):
        self.buffer.add(obs_window, action, reward, done, value, logp)

    def update(self):
        cfg     = self.cfg
        buffer  = self.buffer
        N       = cfg.num_agents

        # ----- bootstrap value ------------------------------------
        with torch.no_grad():
            last_obs = torch.cat([buffer.obs[buffer.ptr - 1], buffer.actions[buffer.ptr - 1]], dim = -1)             # [N,L,D+A]
            if self.cfg.actor_critic_type == "MLP_based":
                last_val = self.critic(last_obs[:,-1].unsqueeze(1))
            else:
                last_val = self.critic(last_obs.unsqueeze(1))

        buffer.compute_returns_adv(
            last_values=last_val,
            gamma=cfg.gamma,
            lam=cfg.gae_lambda
        )

        # ==========================================================
        for _ in range(cfg.ppo_epochs):
            for batch in buffer.get_batches(cfg.batch_size):

                obs_b = batch['obs'].detach()                             # [B,N,L,D]
                acts  = batch['actions'].detach()                         # [B,N,L,A]
                oldlp = batch['logp_old'].detach()                        # [B,N]
                adv   = batch['adv'].detach()                             # [B,N]
                ret   = batch['returns'].detach()
                val_o = batch['values'].detach()

                # ---------- Actor forward -------------------------
                if cfg.actor_critic_type == 'MLP_based':
                    obs_b = obs_b[:, :, -1]
                    obs_list = [obs_b[:, i] for i in range(N)]
                    acts = acts[:, :, -1]
                else:
                    obs_list = [obs_b[:, i] for i in range(N)]

                if cfg.actor_critic_type == 'Recurrent_based':
                    means, stds, _ = self.actor(obs_list, None)
                else:
                    means, stds = self.actor(obs_list)

                means = torch.stack(means)                       # [N,B,A]
                stds  = torch.stack(stds)

                # ---------- Log-prob + entropy --------------------
                logp_new, entropy = [], []
                for i in range(N):
                    dist = torch.distributions.Normal(means[i], stds[i])
                    lp   = dist.log_prob(acts[:, i]).sum(-1, keepdim=True)
                    logp_new.append(lp)
                    entropy.append(dist.entropy().mean())

                logp_new = torch.stack(logp_new)                 # [N,B,1]
                entropy  = torch.stack(entropy).mean()

                # ---------- PPO objective -------------------------
                ratios = torch.exp(logp_new.squeeze(-1) - oldlp.transpose(0,1))
                adv_t  = adv.transpose(0,1)

                surr1 = ratios * adv_t
                surr2 = torch.clamp(ratios, 1-cfg.clip, 1+cfg.clip) * adv_t
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss -= cfg.entropy_coef * entropy

                # ---------- Critic --------------------------------
                if cfg.actor_critic_type == 'MLP_based':
                    critic_in = torch.cat([obs_b, acts],dim = -1).permute(1,0,2)
                else:
                    critic_in = torch.cat([obs_b, acts],dim = -1).permute(1,0,2,3)

                if cfg.actor_critic_type == 'Recurrent_based':
                    new_vals, _ = self.critic(critic_in)
                else:
                    new_vals = self.critic(critic_in)

                v1 = (new_vals - ret).pow(2)
                v2 = (val_o + torch.clamp(new_vals - val_o, -cfg.clip, cfg.clip) - ret).pow(2)
                critic_loss = 0.5 * torch.max(v1, v2).mean()

                # ---------- Update --------------------------------
                loss = actor_loss + cfg.value_coef * critic_loss
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

        buffer.clear()




class MultiAgentDDPG(BaseAlgorithm):
    def __init__(self, actor, critic, cfg, device):
        super().__init__(actor, critic, cfg, device)

        self.actor_target = copy.deepcopy(actor).to(device)
        self.critic_target = copy.deepcopy(critic).to(device)

        self.actor_optim  = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)


        self.replay_buffer = ReplayBuffer(
            capacity=cfg.replay_buffer_size,
            n_agents=cfg.num_agents,
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            sequence_length=cfg.sequence_length,
            device=device
        )
        self.N = cfg.num_agents


    def select_action(self, obs_window, deterministic=False):
        obs_list = _build_actor_input(obs_window, self.cfg.actor_critic_type)

        with torch.no_grad():
            if self.cfg.actor_critic_type == "Recurrent_based":
                actions, _ = self.actor(obs_list, None)
            else:
                actions = self.actor(obs_list)

        actions = torch.stack(actions).squeeze(1)  

        if not deterministic:
            actions += torch.randn_like(actions) * self.cfg.exploration_noise

        return actions

    # ==============================================================
    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    # ==============================================================
    def soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.cfg.tau * p.data + (1.0 - self.cfg.tau) * tp.data)

    # ==============================================================
    # Training update
    # ==============================================================
    def update(self):
        if len(self.replay_buffer) < self.cfg.batch_size:
            return

        obs, act, rew, next_obs, done = self.replay_buffer.sample(self.cfg.batch_size)


        # ----------------------------------------------------------
        # Build critic inputs
        # ----------------------------------------------------------
        if self.cfg.actor_critic_type == "MLP_based":
            obs_c      = obs[:, :, -1, :]          # [B, N, D]
            next_obs_c = next_obs[:, :, -1, :]
            act_c      = act[:, :, -1, :]
        else:
            obs_c      = obs                       # [B, N, L, D]
            next_obs_c = next_obs
            act_c = act

        obs_list = [obs_c[:, i] for i in range(self.N)]
        next_obs_list = [next_obs_c[:, i] for i in range(self.N)]

        # ----------------------------------------------------------
        # Target actions (actor_target)
        # ----------------------------------------------------------

        with torch.no_grad():
            if self.cfg.actor_critic_type == "Recurrent_based":
                next_actions, _ = self.actor_target(next_obs_list, None)
            else:
                next_actions = self.actor_target(next_obs_list)

            next_act = torch.stack(next_actions, dim=1)  

            if self.cfg.actor_critic_type == 'MLP_based':
                critic_target_input = torch.cat([next_obs_c, next_act], dim=-1).permute(1,0,2)
            else:
                critic_target_input = torch.cat([next_obs_c, next_act], dim=-1).permute(1,0,2,3) 
  
            target_q = self.critic_target(critic_target_input)
            done_any = done.any(dim=1, keepdim=True).float()
            y = rew.sum(dim=1, keepdim=True) + self.cfg.gamma * (1.0 - done_any) * target_q


        # ----------------------------------------------------------
        # Critic update
        # ----------------------------------------------------------
        if self.cfg.actor_critic_type == 'MLP_based':
            critic_input = torch.cat([obs_c, act_c], dim=-1).permute(1,0,2)
        else:
            critic_input = torch.cat([obs_c, act_c], dim=-1).permute(1,0,2,3)
        q = self.critic(critic_input)
        critic_loss = nn.MSELoss()(q, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ----------------------------------------------------------
        # Actor update
        # ----------------------------------------------------------

        new_actions = self.actor(obs_list)
        new_actions = torch.stack(new_actions, dim=1)

        if self.cfg.actor_critic_type == 'MLP_based':
            critic_input = torch.cat([obs_c, new_actions], dim = -1).permute(1,0,2)
        else:
            critic_input = torch.cat([obs_c, new_actions], dim = -1).permute(1,0,2,3)  

        actor_loss = -self.critic(critic_input).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ----------------------------------------------------------
        # Target networks
        # ----------------------------------------------------------
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)







class AlgorithmFactory:
    @staticmethod
    def create(name, actor, critic, cfg, device):
        if name == "PPO":
            return MultiAgentPPO(actor, critic, cfg, device)
        elif name == "DDPG":
            return MultiAgentDDPG(actor, critic, cfg, device)
        else:
            raise ValueError(f"Unknown algorithm: {name}")
