import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D


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
    plt.show(block=False)  
    plt.pause(1) 
    plt.close() 

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
    plt.show(block=False)  
    plt.pause(1)  
    plt.close()  

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
    plt.show(block=False)  
    plt.pause(1)  
    plt.close()

def plot_hyperparameter_sweep(param_name, param_values, method, total_timesteps, save_root):
    """
    Loads and plots the results of a hyperparameter sweep.
    """
    results = {}
    for value in param_values:
        tag = f"{method}_{param_name.replace(' ', '').lower()}{value}"
        file_path = os.path.join(save_root, f"results_{tag}_final.npz")
        if os.path.exists(file_path):
            data = np.load(file_path)
            results[tag] = {
                'rewards': data['smoothd_reward'],
                'collisions': data['smoothd_collisions'],
                'success': data['smoothd_success']
            }
        else:
            print(f"Warning: Data file not found for {param_name} = {value} at {file_path}")

    plots_info = [
        ("rewards", f"Reward vs {param_name}", f"comparison_reward_{param_name.replace(' ', '').lower()}"),
        ("collisions", f"Smoothed collision count vs {param_name}", f"comparison_collisions_{param_name.replace(' ', '').lower()}"),
        ("success", f"Smoothed success count vs {param_name}", f"comparison_success_{param_name.replace(' ', '').lower()}")
    ]

    max_x_value_millions = total_timesteps / 1e6

    for metric_key, ylabel, fname in plots_info:
        plt.figure(figsize=(6, 4), dpi=300)
        for value in param_values:
            tag = f"{method}_{param_name.replace(' ', '').lower()}{value}"
            if tag in results:
                y_data = results[tag][metric_key]
                num_points = len(y_data)
                if num_points > 0:
                    x_data = np.linspace(total_timesteps / num_points, total_timesteps, num_points) / 1e6
                    plt.plot(x_data, y_data, label=f"{param_name}: {value}")
        plt.xlabel("Environment steps (millions)")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3, ls='--', lw=0.5)
        plt.xlim(0, max_x_value_millions)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_root, f"{fname}.png"), dpi=300, bbox_inches="tight")
        plt.show(block=False)  
        plt.pause(1)  
        plt.close()

def plot_trajectories(env, trajectories, actor_critic_type, global_step, save_root):
    """
    Plots and saves the 3D trajectories of the agents.
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot cylindrical obstacles
    theta = np.linspace(0, 2 * np.pi, 30)
    z_vals = np.linspace(0, env.grid_size, 2)
    theta, z_mesh = np.meshgrid(theta, z_vals)
    for ox, oy, _ in env.obstacles:
        x_cyl = ox + env.a * np.cos(theta)
        y_cyl = oy + env.a * np.sin(theta)
        ax.plot_surface(x_cyl, y_cyl, z_mesh, color='red', alpha=0.25, linewidth=0)

    # Plot destination spheres (wire-frame)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    u, v = np.meshgrid(u, v)
    
    for i, traj in enumerate(trajectories):
        traj = np.array(traj)
        
        # Trajectory line
        (ln,) = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f'Agent {i}')
        col = ln.get_color()

        # Start marker
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], marker='o', s=50, color=col)

        # Destination sphere
        cx, cy, cz = env.destinations[i]
        r = env.a
        x = cx + r * np.cos(u) * np.sin(v)
        y = cy + r * np.sin(u) * np.sin(v)
        z = cz + r * np.cos(v)
        ax.plot_wireframe(x, y, z, color=col, alpha=0.35)

    # Axis limits & labels
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_zlim(0, env.grid_size)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Generated Trajectories - {actor_critic_type} at step {global_step}')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, f"trajectories_{actor_critic_type}_step{global_step}.png"), dpi=300)
    plt.show(block=False)  
    plt.pause(1)  
    plt.close()
