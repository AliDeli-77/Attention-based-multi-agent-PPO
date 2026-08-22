import torch
import torch.nn as nn
import torch.nn.init as init
import math
from typing import List, Tuple, Optional

class PositionalEncoding(nn.Module):
    """
    Standard sine-cosine positional encoding.
    """
    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:pe[0, :, 1::2].shape[-1]])
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:, :L]

class Attn_MLP_ActorNet(nn.Module):
    """
    Actor with a self-attention encoder followed by an MLP head.
    """
    def __init__(self, input_sizes: List[int], output_sizes: List[int], hidden_mlp: List[int] = [128, 64, 32], n_heads: int = 4, parameter_sharing: bool = True, max_seq_len: int = 512, algo_type = "PPO"):
        super().__init__()
        self.parameter_sharing = parameter_sharing
        self.algo_type = algo_type

        def make_attn(embed_dim):
            assert embed_dim % n_heads == 0, f"embed_dim {embed_dim} not divisible by n_heads {n_heads}"
            return nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)

        if parameter_sharing:
            self.temporal_pool = nn.ModuleList([nn.Linear(input_sizes[0], 1)])
            self.attns = nn.ModuleList([make_attn(input_sizes[0])])
            self.pos_encs = nn.ModuleList([PositionalEncoding(input_sizes[0], max_seq_len)])
            self.log_std = nn.Parameter(torch.zeros(output_sizes[0]))
        else:
            self.temporal_pool = nn.ModuleList([nn.Linear(d, 1) for d in input_sizes])
            self.attns = nn.ModuleList([make_attn(d) for d in input_sizes])
            self.pos_encs = nn.ModuleList([PositionalEncoding(d, max_seq_len) for d in input_sizes])
            self.log_std = nn.ParameterList([nn.Parameter(torch.zeros(out)) for out in output_sizes])

        self.mlps = nn.ModuleList([self._build_mlp(d, out, hidden_mlp) for d, out in zip(input_sizes if parameter_sharing else [a.embed_dim for a in self.attns], output_sizes)]) if parameter_sharing else nn.ModuleList([self._build_mlp(inp_dim=attn.embed_dim, out_dim=out, hidden=hidden_mlp) for attn, out in zip(self.attns, output_sizes)])


    @staticmethod
    def _build_mlp(inp_dim: int, out_dim: int, hidden: List[int]) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev = inp_dim
        for h in hidden:
            fc = nn.Linear(prev, h)
            nn.init.orthogonal_(fc.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(fc.bias)
            layers += [fc, nn.ReLU()]
            prev = h
        out = nn.Linear(prev, out_dim)
        nn.init.orthogonal_(out.weight, gain=0.01)
        nn.init.zeros_(out.bias)
        layers.append(out)
        return nn.Sequential(*layers)

    def forward(self, inputs):
        outputs, stds = [], []

        if self.parameter_sharing:
            attn, p_enc, mlp = self.attns[0], self.pos_encs[0], self.mlps[0]
            log_s = self.log_std

            for seq in inputs:
                seq = p_enc(seq)
                enc, _ = attn(seq, seq, seq)

                weights = torch.softmax(
                    self.temporal_pool[0](enc),
                    dim=1
                )                                    # [B,L,1]

                context = torch.sum(
                    weights * enc,
                    dim=1
                )                                    # [B,D]

                a = mlp(context)

                if self.algo_type == "DDPG":
                    outputs.append(torch.tanh(a))
                else:
                    outputs.append(a)
                    stds.append(log_s.exp().expand_as(a))
        else:
            for seq, attn, p_enc, mlp, log_s, pool in zip(
                inputs, self.attns, self.pos_encs, self.mlps, self.log_std, self.temporal_pool
            ):
                seq = p_enc(seq)
                enc, _ = attn(seq, seq, seq)
                weights = torch.softmax(
                    pool(enc),
                    dim=1
                )

                context = torch.sum(
                    weights * enc,
                    dim=1
                )

                a = mlp(context)

                if self.algo_type == "DDPG":
                    outputs.append(torch.tanh(a))
                else:
                    outputs.append(a)
                    stds.append(log_s.exp().expand_as(a))

        return outputs if self.algo_type == "DDPG" else (outputs, stds)


class Attn_CriticNet(nn.Module):
    """
    Centralised critic using self-attention over the temporal axis.
    """
    def __init__(self, obs_dim: int, act_dim: int, num_agents: int, layers: List[int] = [128, 64, 32], n_heads: int = 4, max_seq_len: int = 512):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.total_obs_dim = num_agents * obs_dim
        assert self.total_obs_dim % n_heads == 0, f"total_obs_dim {self.total_obs_dim} not divisible by n_heads {n_heads}"

        self.attn = nn.MultiheadAttention(embed_dim=self.total_obs_dim, num_heads=n_heads, batch_first=True)
        self.pos_enc = PositionalEncoding(self.total_obs_dim, max_seq_len)
        self.temporal_pool = nn.Linear(self.total_obs_dim, 1)

        mlp_layers: List[nn.Module] = []
        prev = self.total_obs_dim
        for h in layers:
            fc = nn.Linear(prev, h)
            nn.init.orthogonal_(fc.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(fc.bias)
            mlp_layers += [fc, nn.ReLU()]
            prev = h
        out = nn.Linear(prev, num_agents)
        nn.init.orthogonal_(out.weight, gain=1.0)
        nn.init.zeros_(out.bias)
        mlp_layers.append(out)
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        N, B, L, D = states.shape
        assert N == self.num_agents and D == (self.obs_dim + self.act_dim), f"shape mismatch: expected ({self.num_agents},*,*,{self.obs_dim})"
        seq = states.permute(1, 2, 0, 3).contiguous().view(B, L, -1)
        seq = self.pos_enc(seq)
        enc, _ = self.attn(seq, seq, seq)
        weights = torch.softmax(self.temporal_pool(enc),dim=1)                               # [B,L,1]
        context = torch.sum(weights * enc, dim=1)                               # [B,D]
        values = self.mlp(context)
        return values

def _zero_state(n_layers: int, batch: int, hidden: int, device):
    return torch.zeros(n_layers, batch, hidden, device=device)

class GRU_MLP_ActorNet(nn.Module):
    """
    GRU encoder followed by an MLP that outputs Normal-policy parameters.
    """
    def __init__(self, input_sizes: List[int], output_sizes: List[int], hidden_mlp: List[int] = (128, 64, 32), gru_hidden_size: int = 128, num_gru_layers: int = 1, parameter_sharing: bool = True, algo_type = 'PPO'):
        super().__init__()
        self.share = parameter_sharing
        self.H, self.L = gru_hidden_size, num_gru_layers
        self.algo_type = algo_type

        def make_gru(inp):
            g = nn.GRU(inp, gru_hidden_size, num_gru_layers, batch_first=True)
            for n, p in g.named_parameters():
                if "weight" in n: nn.init.orthogonal_(p)
            return g

        self.grus = nn.ModuleList([make_gru(input_sizes[0])] if self.share else [make_gru(d) for d in input_sizes])

        def make_mlp(out_dim):
            layers, prev = [], gru_hidden_size
            for h in hidden_mlp:
                l = nn.Linear(prev, h); nn.init.orthogonal_(l.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(l.bias); layers += [l, nn.ReLU()]; prev = h
            out = nn.Linear(prev, out_dim); nn.init.orthogonal_(out.weight, gain=0.01); nn.init.zeros_(out.bias)
            layers.append(out); return nn.Sequential(*layers)

        self.mlps = nn.ModuleList([make_mlp(output_sizes[0])] if self.share else [make_mlp(o) for o in output_sizes])

        if self.share:
            self.log_std = nn.Parameter(torch.zeros(output_sizes[0]))
        else:
            self.log_std = nn.ParameterList([nn.Parameter(torch.zeros(o)) for o in output_sizes])

    def init_hidden(self, n_agents: int = 1, batch_size: int = 1, device: Optional[torch.device] = None):
        dev = device or next(self.parameters()).device
        base = _zero_state(self.L, batch_size, self.H, dev)
        if self.share:
            return [base.clone() for _ in range(n_agents)]
        else:
            return [base.clone() for _ in range(len(self.grus))]

    @staticmethod
    def detach_hidden(hxs):
        return [h.detach() for h in hxs]

    def forward(self, inputs, hiddens=None):
        if hiddens is None:
            hiddens = self.init_hidden(
                n_agents=len(inputs),
                batch_size=inputs[0].size(0),
                device=inputs[0].device
            )

        outputs, stds, next_h = [], [], []

        if self.share:
            gru, mlp, logS = self.grus[0], self.mlps[0], self.log_std
            for x, h in zip(inputs, hiddens):
                y, h2 = gru(x, h)
                a = mlp(y[:, -1])

                if self.algo_type == "DDPG":
                    outputs.append(torch.tanh(a))
                else:
                    outputs.append(a)
                    stds.append(logS.exp().expand_as(a))

                next_h.append(h2)
        else:
            for x, gru, mlp, logS, h in zip(inputs, self.grus, self.mlps, self.log_std, hiddens):
                y, h2 = gru(x, h)
                a = mlp(y[:, -1])

                if self.algo_type == "DDPG":
                    outputs.append(torch.tanh(a))
                else:
                    outputs.append(a)
                    stds.append(logS.exp().expand_as(a))

                next_h.append(h2)

        return (
            outputs if self.algo_type == "DDPG"
            else (outputs, stds, next_h)
        )


class GRU_CriticNet(nn.Module):
    """
    Centralised recurrent critic.
    """
    def __init__(self, obs_dim:int, act_dim:int, num_agents:int, layers=(128,64,32), gru_hidden=128, gru_layers=1):
        super().__init__(); self.N, self.D = num_agents, obs_dim
        self.act_dim = act_dim
        self.H, self.L = gru_hidden, gru_layers
        self.gru = nn.GRU(obs_dim*num_agents, gru_hidden, gru_layers, batch_first=True)
        for n,p in self.gru.named_parameters():
            if "weight" in n: nn.init.orthogonal_(p)
        mlp, prev = [], gru_hidden
        for h in layers:
            l = nn.Linear(prev, h); nn.init.orthogonal_(l.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(l.bias); mlp += [l, nn.ReLU()]; prev = h
        out = nn.Linear(prev, num_agents); nn.init.orthogonal_(out.weight); nn.init.zeros_(out.bias); mlp.append(out)
        self.mlp = nn.Sequential(*mlp)

    def init_hidden(self, batch=1, device=None):
        dev = device or next(self.parameters()).device
        return _zero_state(self.L, batch, self.H, dev)

    @staticmethod
    def detach_hidden(hx):
        return hx.detach()

    def forward(self, seq: torch.Tensor, hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        N, B, L, D = seq.shape
        assert N == self.N and D == (self.D + self.act_dim), "GRU_CriticNet: shape mismatch"
        if hx is None: hx = self.init_hidden(B, seq.device)
        flat = seq.permute(1,2,0,3).reshape(B, L, N*D)
        y, hx2 = self.gru(flat, hx)
        v = self.mlp(y[:, -1])
        return v, hx2

class CriticMLPNet(nn.Module):
    """
    Centralized Critic network with orthogonal initialization.
    """
    def __init__(self, obs_dim: int, act_dim: int, num_agents: int, layers: List[int] =[128, 64, 32]):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.total_obs_dim = (obs_dim + act_dim) * num_agents
        mlp_layers = []
        prev_dim = self.total_obs_dim
        for h_dim in layers:
            layer = nn.Linear(prev_dim, h_dim)
            init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
            init.zeros_(layer.bias)
            mlp_layers.append(layer)
            mlp_layers.append(nn.ReLU())
            prev_dim = h_dim
        output_layer = nn.Linear(prev_dim, num_agents)
        init.orthogonal_(output_layer.weight, gain=1.0)
        init.zeros_(output_layer.bias)
        mlp_layers.append(output_layer)
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        N, B, D = observations.shape
        assert N == self.num_agents and D == (self.obs_dim + self.act_dim), "Critic input shape mismatch!"
        states = observations.permute(1, 0, 2).contiguous()
        states = states.view(B, -1)
        values = self.mlp(states)
        return values

class MLP_ActorNet(nn.Module):
    def __init__(
        self,
        input_sizes,
        output_sizes,
        hidden=[128, 64, 32],
        parameter_sharing=True,
        algo_type="PPO",          # <-- NEW
        action_limit=1.0
    ):
        super().__init__()
        self.parameter_sharing = parameter_sharing
        self.algo_type = algo_type
        self.action_limit = action_limit

        self.nets = nn.ModuleList(
            [self._build_mlp(inp, out, hidden)
             for inp, out in zip(input_sizes, output_sizes)]
        )

        if self.algo_type == "PPO":
            if parameter_sharing:
                self.log_std = nn.Parameter(torch.zeros(output_sizes[0]))
            else:
                self.log_std = nn.ParameterList(
                    [nn.Parameter(torch.zeros(out)) for out in output_sizes]
                )

    @staticmethod
    def _build_mlp(inp_dim, out_dim, hidden):
        layers, prev = [], inp_dim
        for h in hidden:
            l = nn.Linear(prev, h)
            nn.init.orthogonal_(l.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(l.bias)
            layers += [l, nn.ReLU()]
            prev = h
        out = nn.Linear(prev, out_dim)
        nn.init.orthogonal_(out.weight, gain=0.01)
        nn.init.zeros_(out.bias)
        layers.append(out)
        return nn.Sequential(*layers)

    def forward(self, inputs):
        outputs, stds = [], []

        if self.parameter_sharing:
            net = self.nets[0]
            for obs in inputs:
                a = net(obs)
                if self.algo_type == "DDPG":
                    outputs.append(torch.tanh(a) * self.action_limit)
                else:
                    outputs.append(a)
                    stds.append(self.log_std.exp().expand_as(a))
        else:
            for i, obs in enumerate(inputs):
                a = self.nets[i](obs)
                if self.algo_type == "DDPG":
                    outputs.append(torch.tanh(a) * self.action_limit)
                else:
                    outputs.append(a)
                    stds.append(self.log_std[i].exp().expand_as(a))

        return outputs if self.algo_type == "DDPG" else (outputs, stds)



class CriticMLPNet_DDPG(nn.Module):
    """
    Centralized Critic network with orthogonal initialization.
    """
    def __init__(self, obs_dim: int, num_agents: int, layers: List[int] =[64,128,256,128, 64, 32]):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.total_obs_dim = obs_dim * num_agents
        mlp_layers = []
        prev_dim = self.total_obs_dim
        for h_dim in layers:
            layer = nn.Linear(prev_dim, h_dim)
            init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
            init.zeros_(layer.bias)
            mlp_layers.append(layer)
            mlp_layers.append(nn.ReLU())
            prev_dim = h_dim
        output_layer = nn.Linear(prev_dim, num_agents)
        init.orthogonal_(output_layer.weight, gain=1.0)
        init.zeros_(output_layer.bias)
        mlp_layers.append(output_layer)
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, states: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        N, B, D = states.shape
        assert N == self.num_agents and D == self.obs_dim, "Critic input shape mismatch!"
        states = states.permute(1, 0, 2).contiguous()
        states = states.view(B, -1)
        values = self.mlp(states)
        return values
