import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_rewards(rewards, eps=1e-8):
    """Standaryzacja nagród w batchu: zero mean, unit variance."""
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)


class QLearner:
    def __init__(self, agents, mixer, target_agents, target_mixer, optimizer, gamma, device,
                 grad_clip=10.0, polyak_tau=0.005):
        self.agents = agents
        self.mixer = mixer
        self.target_agents = target_agents
        self.target_mixer = target_mixer
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device
        self.grad_clip = grad_clip
        # Polyak averaging: tau=1.0 to twarda kopia, małe tau (~0.005) to wolne śledzenie
        self.polyak_tau = polyak_tau

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

        # [ZMIANA 1] Standaryzacja nagród przed obliczaniem celu
        rewards = normalize_rewards(rewards)

        B, N, _ = obs.shape

        # 1. Obecne wartości Q dla wybranych akcji
        agent_qs = []
        for i in range(N):
            agent = self.agents[i]
            if hasattr(agent, "rnn"):
                q_vals, _ = agent(obs[:, i, :], hiddens[:, i, :].unsqueeze(0).contiguous())

                # q_vals, _ = agent(obs[:, i, :], hiddens[:, i, :].unsqueeze(0))
            else:
                q_vals = agent(obs[:, i, :])
            chosen_q = q_vals.gather(1, actions[:, i:i + 1]).squeeze(1)
            agent_qs.append(chosen_q)
        agent_qs = torch.stack(agent_qs, dim=1)  # (B, N)

        q_tot = self.mixer(agent_qs, states)      # (B, 1)

        # 2. Double DQN: sieć online wybiera akcję, target network ocenia jej wartość
        with torch.no_grad():
            # [ZMIANA 5] Selekcja akcji przez ONLINE agents (Double DQN)
            online_best_actions = []
            for i in range(N):
              
                agent = self.agents[i]
                if hasattr(agent, "rnn"):
                    online_next_q, _ = agent(next_obs[:, i, :], next_hiddens[:, i, :].unsqueeze(0).contiguous())

                    # online_next_q, _ = agent(next_obs[:, i, :], next_hiddens[:, i, :].unsqueeze(0))
                else:
                    online_next_q = agent(next_obs[:, i, :])
                best_action = online_next_q.argmax(dim=1, keepdim=True)  # (B, 1)
                online_best_actions.append(best_action)

            # Ewaluacja wybranych akcji przez TARGET agents
            target_agent_qs = []
            for i in range(N):
                target_agent = self.target_agents[i]
                if hasattr(target_agent, "rnn"):
                    target_q_vals, _ = target_agent(next_obs[:, i, :], next_hiddens[:, i, :].unsqueeze(0).contiguous())
                    # target_q_vals, _ = target_agent(next_obs[:, i, :], next_hiddens[:, i, :].unsqueeze(0))
                else:
                    target_q_vals = target_agent(next_obs[:, i, :])
                # Zbieramy Q-value dla akcji wybranej przez sieć online
                double_dqn_q = target_q_vals.gather(1, online_best_actions[i]).squeeze(1)
                target_agent_qs.append(double_dqn_q)

            target_agent_qs = torch.stack(target_agent_qs, dim=1)
            target_q_tot = self.target_mixer(target_agent_qs, next_states)  # (B, 1)

        # 3. Równanie Bellmana (TD Target) z Double DQN
        targets = rewards + self.gamma * (1 - dones) * target_q_tot

        # 4. Obliczanie błędu (Loss) i aktualizacja wag
        loss = F.mse_loss(q_tot, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        for param_group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=self.grad_clip)
        self.optimizer.step()

        # [ZMIANA 3] Polyak averaging po każdym kroku treningu
        self._polyak_update()

        return loss.item()

    def _polyak_update(self):
        """Miękka aktualizacja target networks: θ_target = τ*θ_online + (1-τ)*θ_target"""
        tau = self.polyak_tau
        # Aktualizacja agentów (bez duplikatów przy shared weights)
        updated_ids = set()
        for agent, target_agent in zip(self.agents, self.target_agents):
            if id(target_agent) not in updated_ids:
                for p_online, p_target in zip(agent.parameters(), target_agent.parameters()):
                    p_target.data.mul_(1 - tau).add_(tau * p_online.data)
                updated_ids.add(id(target_agent))
        # Aktualizacja mixera
        for p_online, p_target in zip(self.mixer.parameters(), self.target_mixer.parameters()):
            p_target.data.mul_(1 - tau).add_(tau * p_online.data)

    def update_targets(self):
        """Zachowane dla kompatybilności wstecznej — przy polyak averaging nie jest już wywoływane."""
        self._polyak_update()

    def update_bptt(self, batch):
        obs = batch['obs'].to(self.device)         # (B, T, N, Obs)
        states = batch['states'].to(self.device)   # (B, T, State)
        actions = batch['actions'].to(self.device) # (B, T, N)
        rewards = batch['rewards'].to(self.device) # (B, T, 1)
        next_obs = batch['next_obs'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)     # (B, T, 1)
        mask = batch['mask'].to(self.device)       # (B, T, 1)

        # [ZMIANA 1] Standaryzacja nagród — tylko po prawdziwych krokach (maskowanych)
        # Liczymy statystyki tylko z kroków gdzie maska=1, reszta i tak jest zerowana przy loss
        masked_rewards = rewards * mask
        valid_count = mask.sum()
        if valid_count > 1:
            mean = masked_rewards.sum() / valid_count
            # Wariancja liczona tylko po prawdziwych krokach
            sq_diff = ((rewards - mean) ** 2) * mask
            std = (sq_diff.sum() / valid_count).sqrt()
            rewards = (rewards - mean) / (std + 1e-8)

        B, T, N, Obs_Dim = obs.shape
        hidden_dim = getattr(self.agents[0], 'hidden_dim', 128)

        # Sieć online — pełny przebieg przez czas (dla Double DQN potrzebujemy q_vals całości)
        q_vals_list = []
        online_next_q_vals_list = []
        for i in range(N):
            agent = self.agents[i]
            obs_i = obs[:, :, i, :]           # (B, T, Obs)
            next_obs_i = next_obs[:, :, i, :] # (B, T, Obs)
            if hasattr(agent, "rnn"):
                h0_i = torch.zeros(1, B, hidden_dim).to(self.device)
                q_i, h_mid = agent(obs_i, h0_i)
                # [ZMIANA 5] Double DQN: online network ewaluuje next_obs
                online_next_q_i, _ = agent(next_obs_i, h_mid)
            else:
                q_i = agent(obs_i)
                online_next_q_i = agent(next_obs_i)
            q_vals_list.append(q_i)
            online_next_q_vals_list.append(online_next_q_i)

        q_vals = torch.stack(q_vals_list, dim=2)             # (B, T, N, n_actions)
        chosen_q = q_vals.gather(3, actions.unsqueeze(-1)).squeeze(-1)  # (B, T, N)

        # Online wybiera akcje dla next_obs (Double DQN)
        online_next_q_vals = torch.stack(online_next_q_vals_list, dim=2)  # (B, T, N, n_actions)
        online_best_actions = online_next_q_vals.argmax(dim=3, keepdim=True)  # (B, T, N, 1)

        # Target network ocenia wartość wybranych akcji (Double DQN)
        with torch.no_grad():
            target_q_vals_list = []
            for i in range(N):
                target_agent = self.target_agents[i]
                obs_t0_i = obs[:, 0, i, :].unsqueeze(1)  # (B, 1, Obs)
                next_obs_i = next_obs[:, :, i, :]         # (B, T, Obs)

                if hasattr(target_agent, "rnn"):
                    h0_i = torch.zeros(1, B, hidden_dim).to(self.device)
                    _, h1_target_i = target_agent(obs_t0_i, h0_i)
                    target_q_i, _ = target_agent(next_obs_i, h1_target_i)
                else:
                    target_q_i = target_agent(next_obs_i)
                target_q_vals_list.append(target_q_i)

            target_q_vals = torch.stack(target_q_vals_list, dim=2)  # (B, T, N, n_actions)
            # Zbieramy Q-value dla akcji wybranych przez sieć online
            double_dqn_target_q = target_q_vals.gather(3, online_best_actions).squeeze(-1)  # (B, T, N)

        # Przepuszczenie przez Mixery
        q_tot_flat = self.mixer(chosen_q.reshape(B * T, N), states.reshape(B * T, -1))
        q_tot = q_tot_flat.reshape(B, T, 1)

        with torch.no_grad():
            target_q_tot_flat = self.target_mixer(
                double_dqn_target_q.reshape(B * T, N),
                next_states.reshape(B * T, -1)
            )
            target_q_tot = target_q_tot_flat.reshape(B, T, 1)

        # TD-Error i maskowanie
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

        # [ZMIANA 3] Polyak averaging po każdym kroku treningu
        self._polyak_update()

        return loss.item()