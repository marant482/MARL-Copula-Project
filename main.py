import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Importy z Waszych modułów (zakładając że katalog projektu jest w sys.path)
from modules.agents import MLPAgent
from modules.mixers import QMixMixer
from modules.explorers import GaussianCopulaExplorer
from envs.wrappers import SimpleEnvWrapper

def main():
    # --- Konfiguracja ---
    ENV_ID = "Foraging-8x8-2p-2f-v3"
    N_AGENTS = 2
    N_ACTIONS = 6
    OBS_DIM = 12
    STATE_DIM = OBS_DIM * N_AGENTS
    LR = 5e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Inicjalizacja komponentów ---
    raw_env = gym.make(ENV_ID)
    env = SimpleEnvWrapper(raw_env, N_AGENTS, N_ACTIONS)
    
    # Tworzenie agentów i miksera
    agents = nn.ModuleList([MLPAgent(OBS_DIM, N_ACTIONS).to(device) for _ in range(N_AGENTS)])
    mixer = QMixMixer(N_AGENTS, STATE_DIM).to(device)
    
    optimizer = optim.Adam(list(agents.parameters()) + list(mixer.parameters()), lr=LR)
    explorer = GaussianCopulaExplorer(N_AGENTS, correlation=0.7)
    
    # --- Pętla (szkielet) ---
    obs, state = env.reset()
    eps = 1.0 # W docelowym kodzie dodajcie proces decayingu

    # Przykład generowania 1 kroku
    explore_mask = explorer.should_explore(eps)
    actions = []
    
    for i in range(N_AGENTS):
        if explore_mask[i]:
            actions.append(env.env.action_space[0].sample())
        else:
            with torch.no_grad():
                o_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(device)
                q_vals = agents[i](o_tensor)
                # Tutaj w przyszłości wdrożycie nakładanie action_mask
                actions.append(q_vals.argmax(1).item())

    next_obs, next_state, rewards, done, info = env.step(actions)
    
    print(f"Krok wykonany! Wybrane akcje: {actions}, Nagrody: {rewards}")

    # TODO: Dodać zapisywanie do Replay Buffer
    # TODO: Dodać metodę update() uczącą model

if __name__ == "__main__":
    main()
