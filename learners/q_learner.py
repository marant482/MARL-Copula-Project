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
        
        hiddens = batch['hiddens'].to(self.device)           # (B, N, Hidden)
        next_hiddens = batch['next_hiddens'].to(self.device) # (B, N, Hidden)

        B, N, _ = obs.shape

        # 1. Obecne wartości Q dla wybranych akcji
        agent_qs = []
        for i in range(N):
            agent = self.agents[i]
            if hasattr(agent, "rnn"): # Jeśli to agent RNN, przekaż stan ukryty
                q_vals, _ = agent(obs[:, i, :], hiddens[:, i, :])
            else:
                q_vals = agent(obs[:, i, :])
            chosen_q = q_vals.gather(1, actions[:, i:i+1]).squeeze(1)
            agent_qs.append(chosen_q)
        agent_qs = torch.stack(agent_qs, dim=1) # (B, N)

        q_tot = self.mixer(agent_qs, states)    # (B, 1)

        # 2. Docelowe (Target) wartości Q
        with torch.no_grad():
            target_agent_qs = []
            for i in range(N):
                target_agent = self.target_agents[i]
                if hasattr(target_agent, "rnn"):
                    target_q_vals, _ = target_agent(next_obs[:, i, :], next_hiddens[:, i, :])
                else:
                    target_q_vals = target_agent(next_obs[:, i, :])
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
        for param_group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=self.grad_clip)
        self.optimizer.step()

        return loss.item()

    def update_targets(self):
        for agent, target_agent in zip(self.agents, self.target_agents):
            target_agent.load_state_dict(agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def update_bptt(self, batch):
        obs = batch['obs'].to(self.device)  # (B, T, N, Obs)
        states = batch['states'].to(self.device)  # (B, T, State)
        actions = batch['actions'].to(self.device)  # (B, T, N)
        rewards = batch['rewards'].to(self.device)  # (B, T, 1)
        next_obs = batch['next_obs'].to(self.device)  # (B, T, N, Obs)
        next_states = batch['next_states'].to(self.device)  # (B, T, State)
        dones = batch['dones'].to(self.device)  # (B, T, 1)
        mask = batch['mask'].to(self.device)  # (B, T, 1)

        B, T, N, _ = obs.shape

        # Pobieramy wielkość ukrytego stanu (domyślnie zakładamy 128)
        hidden_dim = getattr(self.agents[0], 'hidden_dim', 128)

        # W BPTT stany ukryte RNN inicjujemy zawsze zerami na początku epizodu (t=0)
        hiddens = torch.zeros(B, N, hidden_dim).to(self.device)
        target_hiddens = torch.zeros(B, N, hidden_dim).to(self.device)

        mac_out = []
        target_mac_out = []

        # Rozwijamy sieć w czasie krok po kroku (Unrolling BPTT)
        for t in range(T):
            agent_qs_t = []
            target_agent_qs_t = []
            for i in range(N):
                agent = self.agents[i]
                target_agent = self.target_agents[i]

                # Aktualne wartości Q
                if hasattr(agent, "rnn"):
                    q_vals, hiddens[:, i, :] = agent(obs[:, t, i, :], hiddens[:, i, :])
                else:
                    q_vals = agent(obs[:, t, i, :])
                chosen_q = q_vals.gather(1, actions[:, t, i:i + 1]).squeeze(1)  # Wybrana akcja (B)
                agent_qs_t.append(chosen_q)

                # Target Q-values
                with torch.no_grad():
                    if hasattr(target_agent, "rnn"):
                        target_q_vals, target_hiddens[:, i, :] = target_agent(next_obs[:, t, i, :],
                                                                              target_hiddens[:, i, :])
                    else:
                        target_q_vals = target_agent(next_obs[:, t, i, :])
                    max_target_q = target_q_vals.max(dim=1)[0]  # Maksymalna wartość docelowa
                    target_agent_qs_t.append(max_target_q)

            mac_out.append(torch.stack(agent_qs_t, dim=1))  # -> (B, N)
            target_mac_out.append(torch.stack(target_agent_qs_t, dim=1))  # -> (B, N)

        # Składanie trajektorii czasowej
        mac_out = torch.stack(mac_out, dim=1)  # (B, T, N)
        target_mac_out = torch.stack(target_mac_out, dim=1)  # (B, T, N)

        # Spłaszczanie na potrzeby Mixera, który zazwyczaj przyjmuje kształt (Batch, N)
        mac_out_flat = mac_out.reshape(B * T, N)
        states_flat = states.reshape(B * T, -1)
        q_tot_flat = self.mixer(mac_out_flat, states_flat)
        q_tot = q_tot_flat.reshape(B, T, 1)  # Przywrócenie czasowego kształtu

        with torch.no_grad():
            target_mac_out_flat = target_mac_out.reshape(B * T, N)
            next_states_flat = next_states.reshape(B * T, -1)
            target_q_tot_flat = self.target_mixer(target_mac_out_flat, next_states_flat)
            target_q_tot = target_q_tot_flat.reshape(B, T, 1)

        # TD Target
        targets = rewards + self.gamma * (1 - dones) * target_q_tot

        # Obliczanie straty z zastosowaniem MASKI (ignorujemy padding z bufora epizodycznego)
        td_error = (q_tot - targets.detach())
        masked_td_error = td_error * mask

        # Loss tylko dla odnotowanych akcji
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        for param_group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=self.grad_clip)
        self.optimizer.step()

        return loss.item()