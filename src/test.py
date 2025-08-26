import torch
from src.utils import _build_actor_input

def test_agents(env, actor_net, test_steps: int = 500, sequence_length: int = 8, actor_critic_type: str = 'attention_based', device='cpu'):
    """
    Tests the trained agents in the environment.
    """
    actor_net.eval()
    n_agents = env.num_agents
    obs_t = torch.tensor(env.reset(), dtype=torch.float32, device=device)
    obs_window = obs_t.unsqueeze(1).repeat(1, sequence_length, 1).clone()

    if actor_critic_type == 'Recurrent_based':
        hx_act = actor_net.init_hidden(n_agents, device=device)
    else:
        hx_act = None

    trajectories = [[] for _ in range(n_agents)]

    for _ in range(test_steps):
        obs_list = _build_actor_input(obs_window, actor_critic_type)
        with torch.no_grad():
            if actor_critic_type == 'Recurrent_based':
                means, _, hx_act = actor_net(obs_list, hx_act)
            else:
                means, _ = actor_net(obs_list)
        actions = torch.stack([m.squeeze(0) for m in means]).cpu().numpy()
        obs, _, _, done = env.step(actions)

        for i in range(n_agents):
            trajectories[i].append(env.pos[i].copy())

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        obs_window = torch.roll(obs_window, shifts=-1, dims=1)
        obs_window[:, -1] = obs_t
        if all(done):
            break

    actor_net.train()
    return trajectories
