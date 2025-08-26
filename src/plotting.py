import matplotlib.pyplot as plt
import numpy as np
import os

def plot_comparison_data(timesteps_dict, rewards_dict, collisions_dict, success_dict, methods, save_root):
    """
    Plots and saves comparison graphs for different PPO variants.
    """
    # Smoothed REWARD comparison
    plt.figure(figsize=(6, 4), dpi=300)
    for m in methods:
        plt.plot(timesteps_dict[m], rewards_dict[m], label=m.replace('_', ' ').title())
    plt.xlabel("Environment steps")
    plt.ylabel("Smoothed episode return")
    plt.title("PPO variants – reward")
    plt.grid(alpha=0.3, ls='--', lw=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(f"{save_root}/comparison_reward.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Smoothed COLLISIONS comparison
    plt.figure(figsize=(6, 4), dpi=300)
    for m in methods:
        plt.plot(timesteps_dict[m], collisions_dict[m], label=m.replace('_', ' ').title())
    plt.xlabel("Environment steps")
    plt.ylabel("Smoothed collision count")
    plt.title("PPO variants – collisions")
    plt.grid(alpha=0.3, ls='--', lw=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(f"{save_root}/comparison_collisions.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Smoothed SUCCESSES comparison
    plt.figure(figsize=(6, 4), dpi=300)
    for m in methods:
        plt.plot(timesteps_dict[m], success_dict[m], label=m.replace('_', ' ').title())
    plt.xlabel("Environment steps")
    plt.ylabel("Smoothed success count")
    plt.title("PPO variants – successes")
    plt.grid(alpha=0.3, ls='--', lw=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(f"{save_root}/comparison_success.png", dpi=300, bbox_inches="tight")
    plt.show()
