import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Optional

from src.Algorithms import AlgorithmFactory
from src.utils import ObsNormalizer
from src.test import test_agents
from src.plotting import plot_trajectories
from src.utils import linear_schedule


def train_agents(
    env,
    actor_net,
    critic_net,
    cfg,
    *,
    device,
    save_root: str = "results",
    tag_suffix: str = ""
):

    os.makedirs(save_root, exist_ok=True)

    
    actor_net.to(device)
    critic_net.to(device)

    # =========================
    # Create Algorithm
    # =========================
    algorithm = AlgorithmFactory.create(
        cfg.algorithm,
        actor_net,
        critic_net,
        cfg,
        device
    )

    # =========================
    # Optional Normalization
    # =========================
    obs_norm = ObsNormalizer(cfg.obs_dim).to(device) if cfg.normalize_obs else None

    lr_sched_actor  = linear_schedule(cfg.actor_lr)
    lr_sched_critic  = linear_schedule(cfg.critic_lr)
    ent_sched = linear_schedule(cfg.entropy_coef)


    # =========================
    # Logging
    # =========================
    
    rewards_log, smoothd_reward = deque(maxlen=20), []
    collisions_log, smoothd_collisions = deque(maxlen=20), []
    success_log, smoothd_success = deque(maxlen=20), []

    
    dist_reward_log, smoothd_dist_reward = deque(maxlen=20), []
    col_penalty_log, smoothd_col_penalty = deque(maxlen=20), []

    global_step = 0
    next_test = 0
    ep_ret = 0.0
    ep_dist_ret = 0
    ep_col_ret = 0 

    # =========================
    # Reset Environment
    # =========================
    obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)
    action = torch.zeros(cfg.num_agents, cfg.act_dim).to(device)

    obs_window = obs.unsqueeze(1).repeat( 1, cfg.sequence_length, 1).clone()
    act_window = action.unsqueeze(1).repeat( 1, cfg.sequence_length, 1).clone()


    if obs_norm is not None:
        obs_norm.update(obs)

    # =========================
    # Main Loop
    # =========================
    while global_step < cfg.total_timesteps:

        # -------- Collect Experience --------
        for _ in range(cfg.rollout_steps):

            if cfg.algorithm == "PPO":
                action, logp = algorithm.select_action(obs_window)
                if env.point_mass:
                    action = torch.clamp(action, -env.max_force, env.max_force)
                else:
                    action[:, :3] = torch.clamp(action[:, :3], -env.max_force, env.max_force)  
                    action[:, 3:] = torch.clamp(action[:, 3:], -env.max_moment, env.max_moment)  
                act_window = torch.roll(act_window, shifts=-1, dims=1)
                act_window[:, -1] = action
                observation_window = torch.cat([obs_window, act_window], dim = -1)
                with torch.no_grad():
                    if cfg.actor_critic_type == "MLP_based":
                            value = algorithm.critic((observation_window[:,-1].unsqueeze(1)))
                    else:                   
                            value = algorithm.critic(observation_window.unsqueeze(1))

                next_obs, reward, reward_components, done = env.step(action.cpu().numpy())

                algorithm.store_transition(
                    obs_window,
                    act_window,
                    logp,
                    torch.tensor(reward, device=device),
                    value.squeeze(0),
                    torch.tensor(done, device=device)
                )

            else:  # DDPG
                action = algorithm.select_action(obs_window)
                if cfg.point_mass:
                    action = torch.clamp(action, -env.max_force, env.max_force)
                else:
                    action[:, :3] = torch.clamp(action[:, :3], -env.max_force, env.max_force)  
                    action[:, 3:] = torch.clamp(action[:, 3:], -env.max_moment, env.max_moment)
                next_obs, reward, reward_components, done = env.step(action.cpu().numpy())
                act_window = torch.roll(act_window, shifts=-1, dims=1)
                act_window[:, -1] = action
                next_obs_window = torch.roll(obs_window, shifts=-1, dims=1)
                next_obs_window[:, -1] = torch.tensor(next_obs, device=device)

                algorithm.store_transition(
                    obs_window.clone(),
                    act_window,
                    torch.tensor(reward, device=device),
                    next_obs_window.clone(),
                    torch.tensor(done, device=device)
)


            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            obs_window = torch.roll(obs_window, shifts=-1, dims=1)
            obs_window[:, -1] = obs

            if obs_norm is not None:
                obs_norm.update(obs)

            ep_ret += float(np.mean(reward))
            ep_dist_ret += float(np.mean(reward_components['distance_to_destination']))
            ep_col_ret += float(np.mean(reward_components['collision_with_agents']))
            global_step += 1

            # if all(done):
            #     rewards_log.append(ep_ret); ep_ret = 0.0
            #     dist_reward_log.append(ep_dist_ret); ep_dist_ret = 0.0
            #     col_penalty_log.append(ep_col_ret); ep_col_ret = 0.0

            #     collisions_log.append(env.collision_count)
            #     success_log.append(env.success_count)

            #     obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)
            #     obs_window = obs.unsqueeze(1).repeat(
            #         1, cfg.sequence_length, 1
            #     ).clone()

            #     if obs_norm is not None:
            #         obs_norm.update(obs)
            #     break
        

            

        rewards_log.append(ep_ret); ep_ret = 0.0
        dist_reward_log.append(ep_dist_ret); ep_dist_ret = 0.0
        col_penalty_log.append(ep_col_ret); ep_col_ret = 0.0
        collisions_log.append(env.collision_count)
        success_log.append(env.success_count)

        if len(rewards_log) == rewards_log.maxlen:
            smoothd_reward.append(sum(rewards_log) / rewards_log.maxlen)
            smoothd_dist_reward.append(sum(dist_reward_log) / dist_reward_log.maxlen)
            smoothd_col_penalty.append(sum(col_penalty_log) / col_penalty_log.maxlen)
            smoothd_collisions.append(sum(collisions_log) / collisions_log.maxlen)
            smoothd_success.append(sum(success_log) / success_log.maxlen)

            rewards_log.clear()
            dist_reward_log.clear()
            col_penalty_log.clear()
            collisions_log.clear()
            success_log.clear()

        # -------- Update Algorithm --------

        algorithm.update()

        progress = global_step / cfg.total_timesteps

        new_actor_lr  = lr_sched_actor(1 - progress)
        new_critic_lr = lr_sched_critic(1 - progress)
        cfg.entropy_coef = ent_sched(1 - progress)

        for pg in algorithm.actor_optim.param_groups:
            pg["lr"] = new_actor_lr

        for pg in algorithm.critic_optim.param_groups:
            pg["lr"] = new_critic_lr

        if cfg.algorithm == "DDPG":
            cfg.tau = cfg.tau * (1.0 - progress)


        obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)
        obs_window = obs.unsqueeze(1).repeat(1, cfg.sequence_length, 1).clone()

        action = torch.zeros(cfg.num_agents, cfg.act_dim).to(device)
        act_window = action.unsqueeze(1).repeat( 1, cfg.sequence_length, 1).clone()

        if obs_norm is not None:
            obs_norm.update(obs)


        

        # -------- Plot & save --------
        if global_step >=next_test:
            print(f"[step {global_step:>8}] ")
            plt.figure(figsize=(6, 4))
            plt.plot(smoothd_reward, "-o")
            plt.xlabel("Rollout #")
            plt.ylabel("Cumulative return")
            plt.title(f"Training ({cfg.algorithm}) @ step {global_step}")
            plt.tight_layout()
            plt.savefig(f"{save_root}/Cumulative_return.png", dpi=300, bbox_inches="tight")
            plt.show(block=False) 
            plt.pause(1) 
            plt.close()  

            trajs = test_agents(
                env,
                actor_net,
                cfg.test_steps,
                cfg.sequence_length,
                cfg.actor_critic_type,
                cfg.algorithm,
                device
            )

            plot_trajectories(
                env,
                trajs,
                cfg.actor_critic_type,
                global_step,
                save_root
            )
            next_test += cfg.test_interval

            tag = cfg.algorithm if tag_suffix == "" else f"{cfg.algorithm}_{tag_suffix}"
            torch.save(actor_net.state_dict(), f"{save_root}/actor_{tag}.pth")
            torch.save(critic_net.state_dict(), f"{save_root}/critic_{tag}.pth")

            np.savez(
                os.path.join(save_root, f"results_{tag}.npz"),
                smoothd_reward = np.asarray(smoothd_reward),
                smoothd_dist_reward = np.asarray(smoothd_dist_reward),
                smoothd_col_penalty = np.asarray(smoothd_col_penalty),
                smoothd_collisions = np.asarray(smoothd_collisions),
                smoothd_success=np.asarray(smoothd_success),
            )

    tag = cfg.algorithm if tag_suffix == "" else f"{cfg.algorithm}_{tag_suffix}"
    torch.save(actor_net.state_dict(), f"{save_root}/actor_{tag}.pth")
    torch.save(critic_net.state_dict(), f"{save_root}/critic_{tag}.pth")

    np.savez(
        os.path.join(save_root, f"results_{tag}.npz"),
        smoothd_reward = np.asarray(smoothd_reward),
        smoothd_dist_reward = np.asarray(smoothd_dist_reward),
        smoothd_col_penalty = np.asarray(smoothd_col_penalty),
        smoothd_collisions = np.asarray(smoothd_collisions),
        smoothd_success=np.asarray(smoothd_success),
    )
    return smoothd_reward
