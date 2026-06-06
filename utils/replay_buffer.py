import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, n_agents, obs_dim, state_dim, hidden_dim=128):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.hidden_dim = hidden_dim
        
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        # Nowe tablice na stany ukryte RNN
        self.hiddens = np.zeros((capacity, n_agents, hidden_dim), dtype=np.float32)
        self.next_hiddens = np.zeros((capacity, n_agents, hidden_dim), dtype=np.float32)

    def push(self, obs, state, actions, reward, next_obs, next_state, done, hiddens=None, next_hiddens=None):
        idx = self.ptr
        self.obs[idx] = obs
        self.states[idx] = state
        self.actions[idx] = actions
        
        global_reward = sum(reward) if isinstance(reward, (list, np.ndarray)) else reward
        self.rewards[idx] = global_reward
        
        self.next_obs[idx] = next_obs
        self.next_states[idx] = next_state
        self.dones[idx] = done

        if hiddens is not None:
            self.hiddens[idx] = hiddens
            self.next_hiddens[idx] = next_hiddens

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # 1. Szukamy indeksów z nagrodami
        reward_idxs = np.where(self.rewards[:self.size].sum(axis=-1) > 0)[0]

        # 2. DYNAMICZNY LIMIT:
        # Maksymalnie chcemy 25% batcha...
        max_rewards_allowed = batch_size // 4

        # ...ale nie pozwalamy na wzięcie więcej, niż mamy UNIKALNYCH nagród w buforze!
        guaranteed_rewards = min(max_rewards_allowed, len(reward_idxs))

        # 3. Mechanizm Priorytetów
        if guaranteed_rewards > 0:
            # Losujemy BEZ zwracania (replace=False), bo wiemy, że mamy wystarczająco unikalnych próbek
            idx_rewards = np.random.choice(reward_idxs, guaranteed_rewards, replace=False)

            # Resztę dopychamy zwykłymi wspomnieniami
            remaining_count = batch_size - guaranteed_rewards
            idx_normal = np.random.choice(self.size, remaining_count, replace=False)

            idxs = np.concatenate((idx_rewards, idx_normal))
            np.random.shuffle(idxs)
        else:
            # Brak nagród w buforze = losujemy klasycznie
            idxs = np.random.choice(self.size, batch_size, replace=False)

        # Zwracamy wycięty batch jako słownik tensorów
        return dict(
            obs=torch.FloatTensor(self.obs[idxs]),
            states=torch.FloatTensor(self.states[idxs]),
            actions=torch.LongTensor(self.actions[idxs]),
            rewards=torch.FloatTensor(self.rewards[idxs]),
            next_obs=torch.FloatTensor(self.next_obs[idxs]),
            next_states=torch.FloatTensor(self.next_states[idxs]),
            dones=torch.FloatTensor(self.dones[idxs])
        )

    def __len__(self):
        return self.size
