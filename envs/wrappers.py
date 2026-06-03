import numpy as np

class BaseEnvWrapper:
    def __init__(self, env):
        self.env = env
        
    def reset(self):
        """Musi zwracać: obs_list, global_state"""
        raise NotImplementedError
        
    def step(self, actions):
        """Musi zwracać: next_obs_list, next_global_state, rewards, done, info"""
        raise NotImplementedError

    def get_avail_actions(self):
        """
        Zwraca maskę niedozwolonych akcji.
        Ważne dla Hallway i SMAC. 1 = akcja dozwolona, 0 = niedozwolona.
        """
        raise NotImplementedError

class SimpleEnvWrapper(BaseEnvWrapper):
    # Prosty wrapper dla środowisk bez masek i bez wbudowanego stanu globalnego (np. LBF)
    def __init__(self, env, n_agents, n_actions):
        super().__init__(env)
        self.n_agents = n_agents
        self.n_actions = n_actions
        
    def reset(self):
        obs, _ = self.env.reset()
        global_state = np.concatenate(obs) # Spłaszczamy lokalne obs
        return obs, global_state
        
    def step(self, actions):
        next_obs, rewards, terminated, truncated, info = self.env.step(actions)
        done = bool(np.any(terminated) or np.any(truncated))
        next_global_state = np.concatenate(next_obs)
        return next_obs, next_global_state, rewards, done, info

    def get_avail_actions(self):
        # Wszystkie akcje zawsze dozwolone
        return np.ones((self.n_agents, self.n_actions))
