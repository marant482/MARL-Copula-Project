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
            if hasattr(agent, "rnn"):
                # DODANO unsqueeze(0) aby uzyskać kształt (1, Batch, Hidden)
                q_vals, _ = agent(obs[:, i, :], hiddens[:, i, :].unsqueeze(0))
            else:
                q_vals = agent(obs[:, i, :])
            chosen_q = q_vals.gather(1, actions[:, i:i + 1]).squeeze(1)
            agent_qs.append(chosen_q)
        agent_qs = torch.stack(agent_qs, dim=1) # (B, N)

        q_tot = self.mixer(agent_qs, states)    # (B, 1)

        # 2. Docelowe (Target) wartości Q
        with torch.no_grad():
            target_agent_qs = []
            for i in range(N):
                target_agent = self.target_agents[i]
                if hasattr(target_agent, "rnn"):
                    # DODANO unsqueeze(0)
                    target_q_vals, _ = target_agent(next_obs[:, i, :], next_hiddens[:, i, :].unsqueeze(0))
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

        B, T, N, Obs_Dim = obs.shape
        hidden_dim = getattr(self.agents[0], 'hidden_dim', 128)

        q_vals_list = []
        for i in range(N):
            agent = self.agents[i]
            obs_i = obs[:, :, i, :]  # (B, T, Obs)
            if hasattr(agent, "rnn"):
                h0_i = torch.zeros(1, B, hidden_dim).to(self.device)
                q_i, _ = agent(obs_i, h0_i)
            else:
                q_i = agent(obs_i)
            q_vals_list.append(q_i)

        q_vals = torch.stack(q_vals_list, dim=2)  # (B, T, N, n_actions)
        chosen_q = q_vals.gather(3, actions.unsqueeze(-1)).squeeze(-1)  # (B, T, N)

        # --- POPRAWKA TARGET RNN (Usunięcie Amnezji Czasowej) ---
        with torch.no_grad():
            target_q_vals_list = []
            for i in range(N):
                target_agent = self.target_agents[i]
                obs_t0_i = obs[:, 0, i, :].unsqueeze(1)  # (B, 1, Obs)
                next_obs_i = next_obs[:, :, i, :]  # (B, T, Obs)

                if hasattr(target_agent, "rnn"):
                    h0_i = torch.zeros(1, B, hidden_dim).to(self.device)
                    _, h1_target_i = target_agent(obs_t0_i, h0_i)
                    target_q_i, _ = target_agent(next_obs_i, h1_target_i)
                else:
                    target_q_i = target_agent(next_obs_i)
                target_q_vals_list.append(target_q_i)

            target_q_vals = torch.stack(target_q_vals_list, dim=2)
            max_target_q = target_q_vals.max(dim=3)[0]  # (B, T, N)
        # --------------------------------------------------------

        # Przepuszczenie przez Mixery
        q_tot_flat = self.mixer(chosen_q.reshape(B * T, N), states.reshape(B * T, -1))
        q_tot = q_tot_flat.reshape(B, T, 1)

        with torch.no_grad():
            target_q_tot_flat = self.target_mixer(max_target_q.reshape(B * T, N), next_states.reshape(B * T, -1))
            target_q_tot = target_q_tot_flat.reshape(B, T, 1)

        # 4. TD-Error i maskowanie
        targets = rewards + self.gamma * (1 - dones) * target_q_tot
        td_error = (q_tot - targets.detach())
        masked_td_error = td_error * mask

        mask_sum = mask.sum()
        if mask_sum > 0:
            loss = (masked_td_error ** 2).sum() / mask_sum
        else:
            loss = (masked_td_error ** 2).sum()

        self.optimizer.zero_grad()
        loss.backward()
        for param_group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=self.grad_clip)
        self.optimizer.step()

        return loss.item()