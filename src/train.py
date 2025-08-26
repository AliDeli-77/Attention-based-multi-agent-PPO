import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Optional
from src.utils import RolloutBufferMA, ObsNormalizer, _build_actor_input, linear_schedule
from src.test import test_agents

def ppo_update(policy_net, value_net, roll_buf: 'RolloutBufferMA', policy_opt, value_opt, clip_val: float, ent_coef: float, vf_coef: float, n_epochs: int, batch_size: int, actor_critic_type: str = 'attention_based', target_kl: float | None = None):
    """
    PPO update function with KL-divergence tracking and early stopping.
    """
    target_kl = target_kl or clip_val
    device, N = roll_buf.device, roll_buf.n_agents
    kl_running_sum, n_mb = 0.0, 0

    for _ in range(n_epochs):
        for batch in roll_buf.get_batches(batch_size):
            obs_b = batch['obs']
            if actor_critic_type == 'MLP_based':
                obs_list = [obs_b[:, i, -1] for i in range(N)]
            else:
                obs_list = [obs_b[:, i] for i in range(N)]

            means, stds = (policy_net(obs_list) if actor_critic_type != 'Recurrent_based' else policy_net(obs_list, None)[:2])
            means, stds = torch.stack(means), torch.stack(stds)

            logp_new, entropy = [], []
            for i in range(N):
                dist = torch.distributions.Normal(means[i], stds[i])
                lp = dist.log_prob(batch['actions'][:, i]).sum(-1, keepdim=True)
                logp_new.append(lp)
                entropy.append(dist.entropy().mean())
            logp_new = torch.stack(logp_new)
            entropy = torch.stack(entropy).mean()

            with torch.no_grad():
                approx_kl = (batch['logp_old'].transpose(0, 1) - logp_new).mean()
                kl_running_sum += approx_kl.item()
                n_mb += 1
                if approx_kl > 1.5 * target_kl:
                    print(f"Early-stop PPO epoch — KL {approx_kl:.4f} > 1.5×target")
                    return kl_running_sum / n_mb

            ratios = torch.exp(logp_new - batch['logp_old'].transpose(0, 1))
            adv = batch['adv'].transpose(0, 1).unsqueeze(-1)
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - clip_val, 1 + clip_val) * adv
            actor_loss = -torch.min(surr1, surr2).mean() - ent_coef * entropy

            if actor_critic_type == 'MLP_based':
                critic_in = obs_b[:, :, -1, :].permute(1, 0, 2)
            else:
                critic_in = obs_b.permute(1, 0, 2, 3)
            new_vals = (value_net(critic_in) if actor_critic_type != 'Recurrent_based' else value_net(critic_in)[0])
            val_loss = (new_vals - batch['returns']).pow(2)
            v_clip = batch['values'] + torch.clamp(new_vals - batch['values'], -clip_val, clip_val)
            val_loss2 = (v_clip - batch['returns']).pow(2)
            critic_loss = 0.5 * torch.max(val_loss, val_loss2).mean()

            loss = actor_loss + vf_coef * critic_loss
            policy_opt.zero_grad(); value_opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
            nn.utils.clip_grad_norm_(value_net.parameters(), 10.0)
            policy_opt.step(); value_opt.step()

    return kl_running_sum / max(1, n_mb)

def train_agents(env, actor_net: nn.Module, critic_net: nn.Module, *, total_timesteps: int = 2_000_000, rollout_steps: int = 2048 * 4, n_epochs: int = 10, mini_batch_size: int = 4096, sequence_length: int = 20, gamma: float = 0.995, lam: float = 0.95, clip_val: float = 0.1, ent_coef: float = 0.01, vf_coef: float = 0.5, lr: float = 1e-3, normalize_reward: bool = False, normalize_obs: bool = False, plot_interval: int = 600_000, test_interval: int = 600_000, test_steps: int = 1024 * 4, actor_critic_type: str = 'attention_based', device: Optional[torch.device] = None, save_root: str = "results"):
    """
    Main training routine for PPO.
    """
    os.makedirs(save_root, exist_ok=True)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor_net.to(device); critic_net.to(device)

    lr_sched = linear_schedule(lr)
    ent_sched = linear_schedule(ent_coef, 0.0)

    opt_pi = torch.optim.Adam(actor_net.parameters(), lr=lr)
    opt_v = torch.optim.Adam(critic_net.parameters(), lr=lr)

    n_agents, obs_dim, act_dim = env.num_agents, env.obs_shape, 6
    buffer = RolloutBufferMA(rollout_steps, obs_dim, act_dim, n_agents, sequence_length, device)

    obs_norm = ObsNormalizer(obs_dim).to(device) if normalize_obs else None
    obs_t = torch.tensor(env.reset(), dtype=torch.float32, device=device)
    if normalize_obs: obs_norm.update(obs_t)
    obs_window = obs_t.unsqueeze(1).repeat(1, sequence_length, 1).clone()

    if actor_critic_type == 'Recurrent_based':
        hx_act = actor_net.init_hidden(n_agents, device=device)
        hx_val = critic_net.init_hidden(device=device)
    else:
        hx_act = hx_val = None

    rewards_log, collisions_log, success_log = deque(maxlen=20), deque(maxlen=20), deque(maxlen=20)
    smoothd_reward, smoothd_collisions, smoothd_success = [], [], []
    global_step, ep_ret, next_plot, next_test = 0, 0.0, plot_interval, test_interval

    while global_step < total_timesteps:
        for _ in range(rollout_steps):
            obs_list = _build_actor_input(obs_window, actor_critic_type)
            with torch.no_grad():
                if actor_critic_type == 'Recurrent_based':
                    means, stds, hx_act = actor_net(obs_list, hx_act)
                else:
                    means, stds = actor_net(obs_list)

            actions, logps = [], []
            for i in range(n_agents):
                dist = torch.distributions.Normal(means[i], torch.clamp(stds[i], 1e-3, 1.0))
                a = dist.sample()
                lp = dist.log_prob(a).sum()
                actions.append(a.squeeze(0)); logps.append(lp.unsqueeze(0))

            actions_t = torch.stack(actions)
            logps_t = torch.stack(logps)

            with torch.no_grad():
                if actor_critic_type == 'attention_based':
                    vals = critic_net(obs_window.unsqueeze(1)).squeeze(0)
                elif actor_critic_type == 'Recurrent_based':
                    vals, hx_val = critic_net(obs_window.unsqueeze(1), hx_val)
                    vals = vals.squeeze(0)
                else:
                    vals = critic_net(obs_window[:, -1].unsqueeze(1)).squeeze(0)

            next_obs, reward, _, done = env.step(actions_t.cpu().numpy())
            if normalize_reward:
                reward = env.normalize_rewards(reward)

            buffer.add(obs_window, actions_t, torch.tensor(reward, dtype=torch.float32, device=device), torch.tensor(done, dtype=torch.float32, device=device), vals, logps_t)

            if actor_critic_type == 'Recurrent_based':
                d = torch.as_tensor(done, dtype=torch.float32, device=device)
                masks = (1.0 - d).view(-1, 1, 1)
                hx_act = [h * masks[i] for i, h in enumerate(hx_act)]
                hx_val = hx_val * masks.mean()

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
            if normalize_obs: obs_norm.update(obs_t)
            obs_window = torch.roll(obs_window, shifts=-1, dims=1)
            obs_window[:, -1] = obs_t

            ep_ret += float(np.mean(reward))
            global_step += 1

            if all(done):
                rewards_log.append(ep_ret); ep_ret = 0.0
                collisions_log.append(env.collision_count)
                success_log.append(env.success_count)
                obs_t = torch.tensor(env.reset(), dtype=torch.float32, device=device)
                if normalize_obs: obs_norm.update(obs_t)
                obs_window = obs_t.unsqueeze(1).repeat(1, sequence_length, 1).clone()
                if actor_critic_type == 'Recurrent_based':
                    hx_act = actor_net.init_hidden(n_agents, device=device)
                    hx_val = critic_net.init_hidden(device=device)

        rewards_log.append(ep_ret); ep_ret = 0.0
        if len(rewards_log) == 20:
            smoothd_reward.append(sum(rewards_log) / 20)
            smoothd_collisions.append(sum(collisions_log) / 20)
            smoothd_success.append(sum(success_log) / 20)
            rewards_log.clear(); collisions_log.clear(); success_log.clear()

        with torch.no_grad():
            if actor_critic_type == 'attention_based':
                last_val = critic_net(obs_window.unsqueeze(1)).squeeze(0)
            elif actor_critic_type == 'Recurrent_based':
                last_val, _ = critic_net(obs_window.unsqueeze(1), hx_val)
                last_val = last_val.squeeze(0)
            else:
                last_val = critic_net(obs_window[:, -1].unsqueeze(1)).squeeze(0)

        buffer.compute_returns_adv(last_val, gamma, lam)

        progress = global_step / total_timesteps
        lr_now = lr_sched(progress)
        ent_now = ent_sched(progress)

        for pg in opt_pi.param_groups: pg['lr'] = lr_now
        for pg in opt_v.param_groups: pg['lr'] = lr_now

        mean_kl = ppo_update(actor_net, critic_net, buffer, opt_pi, opt_v, clip_val, ent_now, vf_coef, n_epochs, mini_batch_size, actor_critic_type, target_kl=clip_val)
        print(f"[step {global_step:>8}]  KL={mean_kl:.4f}  ent_coef={ent_now:.5f}")

        buffer.ptr = buffer.full = 0

        if global_step >= next_plot:
            plt.figure(figsize=(6, 4))
            plt.plot(smoothd_reward, '-o')
            plt.xlabel('Rollout #'); plt.ylabel('Cumulative return')
            plt.title(f'Training @ step {global_step}')
            plt.show(); next_plot += plot_interval

        if global_step >= next_test:
            trajs = test_agents(env, actor_net, test_steps, sequence_length, actor_critic_type, device)
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            theta = np.linspace(0, 2 * np.pi, 30)
            z_vals = np.linspace(0, env.grid_size, 2)
            theta, z_mesh = np.meshgrid(theta, z_vals)

            for ox, oy, _ in env.obstacles:
                x_cyl = ox + env.a * np.cos(theta)
                y_cyl = oy + env.a * np.sin(theta)
                ax.plot_surface(x_cyl, y_cyl, z_mesh, color='red', alpha=0.25, linewidth=0, label='_nolegend_')

            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 15)
            u, v = np.meshgrid(u, v)

            for i, traj in enumerate(trajs):
                traj = np.array(traj)
                (ln,) = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f'Agent {i}')
                col = ln.get_color()
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], marker='o', s=50, color=col)
                cx, cy, cz = env.destinations[i]
                r = env.a
                x = cx + r * np.cos(u) * np.sin(v)
                y = cy + r * np.sin(u) * np.sin(v)
                z = cz + r * np.cos(v)
                ax.plot_wireframe(x, y, z, color=col, alpha=0.35)

            ax.set_xlim(0, env.grid_size); ax.set_ylim(0, env.grid_size); ax.set_zlim(0, env.grid_size)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_title(f'{actor_critic_type}: trajectories @ step {global_step}')
            ax.legend(loc='upper left')
            plt.tight_layout()
            plt.show()

            next_test += test_interval

            tag = f"{actor_critic_type}"
            torch.save(actor_net.state_dict(), f"{save_root}/actor_{tag}.pth")
            torch.save(critic_net.state_dict(), f"{save_root}/critic_{tag}.pth")
            np.savez(os.path.join(save_root, f"results_{tag}.npz"), smoothd_reward=np.asarray(smoothd_reward), smoothd_collisions=np.asarray(smoothd_collisions), smoothd_success=np.asarray(smoothd_success))

    return smoothd_reward, smoothd_collisions, smoothd_success
