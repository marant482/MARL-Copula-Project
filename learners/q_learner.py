import torch
import torch.nn as nn
import torch.nn.functional as F

class QLearner:
    def __init__(self, agents, mixer, target_agents, target_mixer, optimizer, gamma, device, grad_clip=10.0):
        self.agents = agents
        self.mixer = mixer
        self.target_agents = target_agents
        self.target_mixer = target_mixer
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device
        self.grad_clip = grad_clip

    def update(self, batch):
        obs = batch['obs'].to(self.device)             # (B, N, Obs)
        states = batch['states'].to(self.device)       # (B, State)
        actions = batch['actions'].to(self.device)     # (B, N)
        rewards = batch['rewards'].to(self.device)     # (B, 1)
        next_obs = batch['next_obs'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)

        B, N, _ = obs.shape

        # 1. Obecne wartości Q dla wybranych akcji
        agent_qs = []
        for i in range(N):
            q_vals = self.agents[i](obs[:, i, :])
            chosen_q = q_vals.gather(1, actions[:, i:i+1]).squeeze(1)
            agent_qs.append(chosen_q)
        agent_qs = torch.stack(agent_qs, dim=1) # (B, N)

        q_tot = self.mixer(agent_qs, states)    # (B, 1)

        # 2. Docelowe (Target) wartości Q
        with torch.no_grad():
            target_agent_qs = []
            for i in range(N):
                target_q_vals = self.target_agents[i](next_obs[:, i, :])
                max_target_q = target_q_vals.max(dim=1)[0]
                target_agent_qs.append(max_target_q)
            target_agent_qs = torch.stack(target_agent_qs, dim=1)
            target_q_tot = self.target_mixer(target_agent_qs, next_states) # (B, 1)

        # 3. Równanie Bellmana (TD Target)
        targets = rewards + self.gamma * (1 - dones) * target_q_tot

        # 4. Obliczanie błędu (Loss) i aktualizacja wag
        loss = F.mse_loss(q_tot, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        # Przycinanie gradientów, żeby uczenie nie "wybuchło"for param_group in self.optimizer.param_groups:
        for param_group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=self.grad_clip)
        self.optimizer.step()

        return loss.item()

    def update_targets(self):
        # Aktualizacja sieci docelowych (wykonywana co X kroków)
        for agent, target_agent in zip(self.agents, self.target_agents):
            target_agent.load_state_dict(agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
