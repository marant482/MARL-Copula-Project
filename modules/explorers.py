import numpy as np
from scipy.stats import norm
import random

class BaseExplorer:
    def should_explore(self, epsilon):
        raise NotImplementedError

class EpsilonGreedyExplorer(BaseExplorer):
    def __init__(self, n_agents):
        self.n_agents = n_agents

    def should_explore(self, epsilon):
        # Zwraca listę booleanów (True jeśli agent losuje akcję)
        return [random.random() < epsilon for _ in range(self.n_agents)]

class GaussianCopulaExplorer(BaseExplorer):
    def __init__(self, n_agents, correlation=0.7):
        self.n_agents = n_agents
        self.set_correlation(correlation)

    def set_correlation(self, rho):
        # Zależnie od Waszych badań, to rho będzie wektorem/macierzą wyliczaną dynamicznie
        self.Sigma = np.full((self.n_agents, self.n_agents), rho)
        np.fill_diagonal(self.Sigma, 1.0)
        self.Sigma += np.eye(self.n_agents) * 1e-6 # Zabezpieczenie PSD

    def should_explore(self, epsilon):
        u = np.random.multivariate_normal(np.zeros(self.n_agents), self.Sigma)
        p = norm.cdf(u)
        return p < epsilon


class ActionCopulaSampler:
    def __init__(self, n_agents, n_actions, correlation=0.0):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.set_correlation(correlation)

    def set_correlation(self, rho):
        self.Sigma = np.full((self.n_agents, self.n_agents), rho)
        np.fill_diagonal(self.Sigma, 1.0)
        self.Sigma += np.eye(self.n_agents) * 1e-6  # Zabezpieczenie PSD

    def sample(self):
        # 1. Losujemy ze skorelowanego rozkładu normalnego
        z = np.random.multivariate_normal(np.zeros(self.n_agents), self.Sigma)

        # 2. Przekształcamy na rozkład jednostajny [0, 1] za pomocą dystrybuanty (CDF)
        u = norm.cdf(z)

        # 3. Mapujemy jednostajny [0, 1] na dyskretne indeksy akcji [0, 1, ..., n_actions - 1]
        actions = np.floor(u * self.n_actions).astype(int)

        # Zabezpieczenie (np. gdyby u wyszło równe 1.0, co dałoby indeks poza zakresem)
        actions = np.clip(actions, 0, self.n_actions - 1)
        return actions.tolist()