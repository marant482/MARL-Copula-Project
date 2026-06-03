import torch
import torch.nn as nn

class VDNMixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, agent_qs, states=None):
        # agent_qs shape: (batch_size, n_agents)
        # VDN po prostu sumuje wartości Q, nie używa stanu globalnego
        return agent_qs.sum(dim=1, keepdim=True)

class QMixMixer(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Linear(state_dim, embed_dim * n_agents)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        # agent_qs shape: (batch_size, n_agents)
        # states shape: (batch_size, state_dim)
        B = agent_qs.size(0)
        x = agent_qs.unsqueeze(1) # (B, 1, n_agents)

        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).unsqueeze(1)
        h = torch.relu(torch.bmm(x, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states)).unsqueeze(2)
        b2 = self.hyper_b2(states)

        q_tot = torch.bmm(h, w2).squeeze(2) + b2
        return q_tot
