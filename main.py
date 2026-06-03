import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import copy
from tqdm import tqdm

from modules.agents import MLPAgent
from modules.mixers import QMixMixer
from modules.explorers import GaussianCopulaExplorer, EpsilonGreedyExplorer
from envs.wrappers import SimpleEnvWrapper
from utils.replay_buffer import ReplayBuffer
from learners.q_learner import QLearner

import lbforaging # Wymagane by rejestrować środowiska LBF w gym

def main():
    # --- Konfiguracja ---
    ENV_ID = "Foraging-8x8-2p-2f-v3"
    N_AGENTS = 2
    N_ACTIONS = 6
    OBS_DIM = 12
    STATE_DIM = OBS_DIM * N_AGENTS
    
    TOTAL_STEPS = 50_000
    BATCH_SIZE = 32
    BUFFER_SIZE = 10_000
    MIN_BUFFER_SIZE = 500
    TARGET_UPDATE_INTERVAL = 200
    GAMMA = 0.99
    LR = 5e-4
    
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 20_000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używam urządzenia: {device}")

    # --- Inicjalizacja komponentów ---
    raw_env = gym.make(ENV_ID)
    env = SimpleEnvWrapper(raw_env, N_AGENTS, N_ACTIONS)
    
    # Modele główne
    agents = nn.ModuleList([MLPAgent(OBS_DIM, N_ACTIONS).to(device) for _ in range(N_AGENTS)])
    mixer = QMixMixer(N_AGENTS, STATE_DIM).to(device)
    
    # Modele docelowe (Target networks - kopie głównych modeli)
    target_agents = copy.deepcopy(agents)
    target_mixer = copy.deepcopy(mixer)
    
    optimizer = optim.Adam(list(agents.parameters()) + list(mixer.parameters()), lr=LR)
    
    # Tutaj możecie przełączać eksploratora w ramach testów
    explorer = GaussianCopulaExplorer(N_AGENTS, correlation=0.7)
    # explorer = EpsilonGreedyExplorer(N_AGENTS)
    
    buffer = ReplayBuffer(BUFFER_SIZE, N_AGENTS, OBS_DIM, STATE_DIM)
    learner = QLearner(agents, mixer, target_agents, target_mixer, optimizer, GAMMA, device)
    
    # --- Pętla Treningowa ---
    obs, state = env.reset()
    episode_reward = 0
    episodes_done = 0
    
    pbar = tqdm(total=TOTAL_STEPS, desc="Trening QMIX")
    
    for step in range(TOTAL_STEPS):
        # Spadek Epsilona (Decay)
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / EPS_DECAY)
        
        # Wybór akcji
        explore_mask = explorer.should_explore(eps)
        actions = []
        
        for i in range(N_AGENTS):
            if explore_mask[i]:
                actions.append(env.env.action_space[0].sample())
            else:
                with torch.no_grad():
                    o_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(device)
                    q_vals = agents[i](o_tensor)
                    actions.append(q_vals.argmax(1).item())

        # Wykonanie kroku w środowisku
        next_obs, next_state, rewards, done, info = env.step(actions)
        episode_reward += sum(rewards)
        
        # Zapis do pamięci
        buffer.push(obs, state, actions, rewards, next_obs, next_state, float(done))
        
        obs, state = next_obs, next_state
        
        # Uczenie
        if len(buffer) >= MIN_BUFFER_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            loss = learner.update(batch)
            
            # Aktualizacja sieci docelowych
            if step % TARGET_UPDATE_INTERVAL == 0:
                learner.update_targets()
                
        if done:
            episodes_done += 1
            if episodes_done % 10 == 0:
                pbar.set_postfix({"Ostatnia Nagroda Epizodu": episode_reward, "Epsilon": f"{eps:.2f}"})
            
            obs, state = env.reset()
            episode_reward = 0
            
        pbar.update(1)

    pbar.close()
    print("Trening zakończony!")

if __name__ == "__main__":
    main()
