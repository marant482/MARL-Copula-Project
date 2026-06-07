import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, n_agents, obs_dim, state_dim, hidden_dim=128, reward_priority=0.25):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.hidden_dim = hidden_dim
        self.reward_priority = reward_priority

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

        # 2. DYNAMICZNY LIMIT wyliczany z proporcji (np. 0.25 to 25% batcha)
        max_rewards_allowed = int(batch_size * self.reward_priority)

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
            dones=torch.FloatTensor(self.dones[idxs]),
            # TE DWIE LINIJKI SĄ KLUCZOWE DLA RNN:
            hiddens=torch.FloatTensor(self.hiddens[idxs]),
            next_hiddens=torch.FloatTensor(self.next_hiddens[idxs])
        )

    def __len__(self):
        return self.size


class EpisodicReplayBuffer:
    def __init__(self, capacity, max_steps, n_agents, obs_dim, state_dim, reward_priority=0.25):
        self.capacity = capacity
        self.max_steps = max_steps
        self.ptr = 0
        self.size = 0
        self.reward_priority = reward_priority

        # Wymiary: (Pojemność w epizodach, Czas (T), Agenci, Cechy)
        self.obs = np.zeros((capacity, max_steps, n_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, max_steps, n_agents, obs_dim), dtype=np.float32)
        self.states = np.zeros((capacity, max_steps, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, max_steps, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, max_steps, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, max_steps, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, max_steps, 1), dtype=np.float32)
        # Maska do ignorowania paddingu przy wyliczaniu loss
        self.mask = np.zeros((capacity, max_steps, 1), dtype=np.float32)

    def push_episode(self, episode_data):
        idx = self.ptr
        T = min(len(episode_data['obs']), self.max_steps)

        # Czyszczenie maski dla nadpisywanego epizodu
        self.mask[idx] = 0.0

        for t in range(T):
            self.obs[idx, t] = episode_data['obs'][t]
            self.states[idx, t] = episode_data['states'][t]
            self.actions[idx, t] = episode_data['actions'][t]
            self.rewards[idx, t] = episode_data['rewards'][t]
            self.next_obs[idx, t] = episode_data['next_obs'][t]
            self.next_states[idx, t] = episode_data['next_states'][t]
            self.dones[idx, t] = episode_data['dones'][t]
            self.mask[idx, t] = 1.0  # 1.0 oznacza rzeczywisty krok

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # 1. Szukamy indeksów EPIZODÓW, w których padła jakakolwiek nagroda
        # self.rewards ma kształt (capacity, max_steps, 1), więc sumujemy po czasie (axis=1) i wartościach (axis=2)
        episode_rewards = self.rewards[:self.size].sum(axis=(1, 2))
        reward_idxs = np.where(episode_rewards > 0)[0]

        # 2. DYNAMICZNY LIMIT wyliczany z proporcji:
        max_rewards_allowed = int(batch_size * self.reward_priority)

        # ...ale nie pozwalamy na wzięcie więcej, niż mamy UNIKALNYCH epizodów z nagrodą w buforze
        guaranteed_rewards = min(max_rewards_allowed, len(reward_idxs))

        # 3. Mechanizm Priorytetów
        if guaranteed_rewards > 0:
            # Losujemy epizody z nagrodą BEZ zwracania
            idx_rewards = np.random.choice(reward_idxs, guaranteed_rewards, replace=False)

            # Resztę dopychamy zwykłymi (często pustymi) epizodami z całej puli
            remaining_count = batch_size - guaranteed_rewards
            idx_normal = np.random.choice(self.size, remaining_count, replace=False)

            idxs = np.concatenate((idx_rewards, idx_normal))
            np.random.shuffle(idxs)
        else:
            # Brak jakichkolwiek nagród w buforze = losujemy całkowicie klasycznie
            idxs = np.random.choice(self.size, batch_size, replace=False)

        # Zwracamy wycięty batch epizodów jako słownik tensorów
        return dict(
            obs=torch.FloatTensor(self.obs[idxs]),
            states=torch.FloatTensor(self.states[idxs]),
            actions=torch.LongTensor(self.actions[idxs]),
            rewards=torch.FloatTensor(self.rewards[idxs]),
            next_obs=torch.FloatTensor(self.next_obs[idxs]),
            next_states=torch.FloatTensor(self.next_states[idxs]),
            dones=torch.FloatTensor(self.dones[idxs]),
            mask=torch.FloatTensor(self.mask[idxs])
        )

    def __len__(self):
        return self.size