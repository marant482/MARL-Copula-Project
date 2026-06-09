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
    """
    QMix z hipersieciami jako małymi sieciami neuronowymi (zamiast liniowych warstw).
    Każda hypernet to dwuwarstwowa MLP: state_dim -> embed_dim -> output_dim,
    co pozwala generować bardziej ekspresywne wagi zależne od stanu globalnego.
    """
    def __init__(self, n_agents, state_dim, embed_dim=32, hypernet_hidden=64):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        # [ZMIANA 4] Hipersieci jako małe sieci neuronowe (2 warstwy) zamiast liniowych

        # Hypernet dla wag pierwszej warstwy: state -> embed_dim*n_agents
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, embed_dim * n_agents)
        )

        # Hypernet dla biasów pierwszej warstwy: state -> embed_dim
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, embed_dim)
        )

        # Hypernet dla wag drugiej warstwy: state -> embed_dim
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, embed_dim)
        )

        # Hypernet dla biasów drugiej warstwy (ze skalowaniem): state -> 1
        # Zachowuje oryginalną strukturę z nieliniowością (stan -> embed -> 1)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        # agent_qs shape: (batch_size, n_agents)
        # states shape:   (batch_size, state_dim)
        B = agent_qs.size(0)
        x = agent_qs.unsqueeze(1)  # (B, 1, n_agents)

        # Pierwsza warstwa mixera: wagi muszą być nieujemne (monotonicity constraint QMix)
        w1 = torch.abs(self.hyper_w1(states))          # (B, embed_dim * n_agents)
        w1 = w1.view(B, self.n_agents, self.embed_dim) # (B, n_agents, embed_dim)
        b1 = self.hyper_b1(states).unsqueeze(1)        # (B, 1, embed_dim)
        h = torch.relu(torch.bmm(x, w1) + b1)          # (B, 1, embed_dim)

        # Druga warstwa mixera
        w2 = torch.abs(self.hyper_w2(states)).unsqueeze(2)  # (B, embed_dim, 1)
        b2 = self.hyper_b2(states)                          # (B, 1)

        q_tot = torch.bmm(h, w2).squeeze(2) + b2            # (B, 1)
        return q_tot