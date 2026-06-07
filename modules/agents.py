import torch
import torch.nn as nn
import numpy as np

class MLPAgent(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128, num_layers=2):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, n_actions))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        return self.net(obs)


class RNNAgent(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        # BARDZO WAŻNE: Używamy nn.GRU z batch_first=True
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, obs, hidden):
        # Automatyczne wykrywanie, czy to pojedynczy krok czy cały epizod z BPTT
        if len(obs.shape) == 2:
            # Gra (pojedynczy krok): Zmieniamy (Batch, Obs) na (Batch, 1_krok, Obs)
            x = torch.relu(self.fc1(obs)).unsqueeze(1)
            out, hidden = self.rnn(x, hidden)
            q = self.fc2(out.squeeze(1))
        else:
            # Trening (cały epizod): Mamy już wymiar (Batch, Czas, Obs)
            x = torch.relu(self.fc1(obs))
            out, hidden = self.rnn(x, hidden)
            q = self.fc2(out)

        return q, hidden
