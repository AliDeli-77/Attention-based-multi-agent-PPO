import os
import argparse
import torch
from types import SimpleNamespace
import numpy as np
from src.plotting import plot_comparison_data

from src.Environments import MultiAgentEnv
from src.Models import (
    Attn_MLP_ActorNet, Attn_CriticNet,
    GRU_MLP_ActorNet, GRU_CriticNet,
    MLP_ActorNet, CriticMLPNet
)
from src.train import train_agents
from src.test import test_agents
from src.utils import ReplayBuffer


# =====================================================
# Model Factory
# =====================================================

def make_models(actor_critic_type, obs_dim, act_dim, n_agents, algorithm):
    if actor_critic_type == "attention_based":
        actor = Attn_MLP_ActorNet([obs_dim] * n_agents, [act_dim] * n_agents, n_heads=3, algo_type=algorithm)
        critic = Attn_CriticNet(obs_dim, act_dim, n_agents, n_heads=3)

    elif actor_critic_type == "Recurrent_based":
        actor = GRU_MLP_ActorNet([obs_dim] * n_agents, [act_dim] * n_agents, algo_type=algorithm)
        critic = GRU_CriticNet(obs_dim, act_dim, n_agents)

    elif actor_critic_type == "MLP_based":
        actor = MLP_ActorNet([obs_dim] * n_agents, [act_dim] * n_agents, algo_type=algorithm)
        critic = CriticMLPNet(obs_dim, act_dim, n_agents)

    else:
        raise ValueError(f"Unknown actor_critic_type: {actor_critic_type}")

    return actor, critic


def compare_algorithms(cfg, save_root="results"):
 

    methods = ["PPO", "DDPG"]

    rewards_dict = {}
    collisions_dict = {}
    success_dict = {}
    timesteps_dict = {}

    available_methods = []

    for method in methods:
        file_path = f"{save_root}/results_{method}_{cfg.actor_critic_type}.npz"

        if not os.path.exists(file_path):
            print(f"[INFO] Comparison skipped: {file_path} not found.")
            continue

        data = np.load(file_path)

        rewards_dict[method] = data["smoothd_reward"]
        collisions_dict[method] = data["smoothd_collisions"]
        success_dict[method] = data["smoothd_success"]

        n_points = len(rewards_dict[method])

        timesteps_dict[method] = np.linspace(
            cfg.rollout_steps,
            cfg.total_timesteps,
            n_points
        )

        available_methods.append(method)

    if len(available_methods) < 2:
        print("[INFO] Need both PPO and DDPG results for comparison.")
        return

    plot_comparison_data(
        timesteps_dict,
        rewards_dict,
        collisions_dict,
        success_dict,
        available_methods,
        save_root
    )

    print("[INFO] Comparison plots saved successfully.")


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Multi-Agent RL Training")

    parser.add_argument("--algorithm", type=str, default="PPO",
                        choices=["PPO", "DDPG"], help="RL algorithm")

    parser.add_argument("--model_type", type=str, default="attention_based",
                        choices=["attention_based", "Recurrent_based", "MLP_based"])

    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--model_path", type=str,
                    default="results/actor_PPO_attention_based.pth")

    args = parser.parse_args()

    save_root = "results"
    os.makedirs(save_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    # =================================================
    # Environment
    # =================================================
    env = MultiAgentEnv(num_agents=2, grid_size=1, use_quaternion=False)
    obs_dim = env.obs_shape
    n_agents = env.num_agents
    act_dim = 6

    # =================================================
    # Models
    # =================================================
    actor, critic = make_models(
                                    args.model_type,
                                    obs_dim,
                                    act_dim,
                                    n_agents,
                                    args.algorithm
                                )

    # =================================================
    # Plot-only mode
    # =================================================
    if args.plot_only:
        actor.load_state_dict(torch.load(args.model_path, map_location=device))
        actor.to(device)

        print(f"Loaded model from {args.model_path}")
        print("Generating trajectories...")

        trajectories = test_agents(
            env,
            actor,
            test_steps=2000,
            sequence_length=30,
            actor_critic_type=args.model_type,
            algorithm = args.algorithm,
            device = device
        )
        print("Done.")
        exit(0)

    # =================================================
    # Configuration Object
    # =================================================
    cfg = SimpleNamespace(

        # --- General ---
        algorithm = args.algorithm,
        actor_critic_type=args.model_type,
        total_timesteps=10_000_000,
        rollout_steps=2048 * 6,
        obs_dim=obs_dim,
        act_dim = act_dim,
        num_agents=n_agents,
        normalize_obs=False,

        # --- PPO specific ---
        gamma=0.995,
        gae_lambda=0.95,
        clip=0.2,
        ppo_epochs=2,
        batch_size=2048,
        value_coef=0.5,
        entropy_coef=0.01,
        actor_lr=1e-3,
        critic_lr=1e-3,
        max_grad_norm=0.1,

        # --- DDPG specific ---
        tau=0.005,
        exploration_noise=0.0001,
        replay_buffer_size=500000,   

        # --- Evaluation ---
        plot_interval=600_000,
        test_interval=600_000,
        test_steps=4096,
        sequence_length=10,
    )

    

    if args.algorithm == "DDPG":
        cfg.replay_buffer = ReplayBuffer(
            capacity=1_000_000,
            n_agents=n_agents,
            obs_dim=obs_dim,
            act_dim=act_dim,
            sequence_length=10,
            device=device,
        )

    


    # =================================================
    # Train
    # =================================================
    print(f"\nStarting training:")
    print(f"  Algorithm: {cfg.algorithm}")
    print(f"  Model:     {cfg.actor_critic_type}\n")

    train_agents(
        env,
        actor,
        critic,
        cfg,
        device=device,
        save_root=save_root,
        tag_suffix=args.model_type
    )

    compare_algorithms(cfg, save_root)

    
