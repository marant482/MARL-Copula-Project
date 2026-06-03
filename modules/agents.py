import torch
import torch.nn as nn
import numpy as np

class MLPAgent(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128, num_layers=2):
        super().__init__()
        
        # Pierwsza warstwa (wejściowa)
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        
        # Dynamiczne dodawanie warstw ukrytych
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            
        # Warstwa wyjściowa
        layers.append(nn.Linear(hidden_dim, n_actions))
        
        self.net = nn.Sequential(*layers)
        
        # Właściwa inicjalizacja ortogonalna (dla stabilności)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        return self.net(obs)
