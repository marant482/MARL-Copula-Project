import argparse
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

import lbforaging

def parse_args():
    parser = argparse.ArgumentParser(description="Trening MARL z Kopulą Gaussa")
    
    # Parametry środowiska
    parser.add_argument("--env_id", type=str, default="Foraging-8x8-2p-2f-v3", help="ID środowiska Gym")
    parser.add_argument("--n_agents", type=int, default=2, help="Liczba agentów")
    parser.add_argument("--n_actions", type=int, default=6, help="Liczba dostępnych akcji")
    parser.add_argument("--obs_dim", type=int, default=12, help="Wymiar wektora obserwacji")
    
    # Parametry treningu
    parser.add_argument("--mixer", type=str, default="qmix", choices=["qmix", "vdn"], help="Typ miksera (qmix/vdn)")
    parser.add_argument("--total_steps", type=int, default=50000, help="Całkowita liczba kroków w środowisku")
    parser.add_argument("--batch_size", type=int, default=32, help="Rozmiar batcha")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Rozmiar bufora pamięci")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (szybkość uczenia)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Współczynnik dyskontowania (gamma)")
    
    # Parametry eksploracji
    parser.add_argument("--copula_corr", type=float, default=0.7, help="Wartość korelacji (rho) dla Kopuli")
    parser.add_argument("--explorer", type=str, default="copula", choices=["copula", "epsilon"], help="Typ eksploratora (copula/epsilon)")
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=int, default=20000, help="Czas opadania epsilona (w krokach)")
    
    return parser.parse_args()

def main():
    # Pobranie argumentów z wywołania skryptu
    args = parse_args()
    
    STATE_DIM = args.obs_dim * args.n_agents
    TARGET_UPDATE_INTERVAL = 200
    MIN_BUFFER_SIZE = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używam urządzenia: {device}")
    print(f"Start treningu: Środowisko={args.env_id}, Kroki={args.total_steps}, Eksplorator={args.explorer}")

    # Inicjalizacja komponentów
    raw_env = gym.make(args.env_id)
    env = SimpleEnvWrapper(raw_env, args.n_agents, args.n_actions)
    
    agents = nn.ModuleList([MLPAgent(args.obs_dim, args.n_actions).to(device) for _ in range(args.n_agents)])
    if args.mixer == "qmix":
        mixer = QMixMixer(args.n_agents, STATE_DIM).to(device)
    elif args.mixer == "vdn":
        mixer = VDNMixer().to(device)
    
    target_agents = copy.deepcopy(agents)
    target_mixer = copy.deepcopy(mixer)
    
    optimizer = optim.Adam(list(agents.parameters()) + list(mixer.parameters()), lr=args.lr)
    
    # Dynamiczny wybór eksploratora na podstawie argumentu
    if args.explorer == "copula":
        explorer = GaussianCopulaExplorer(args.n_agents, correlation=args.copula_corr)
    else:
        explorer = EpsilonGreedyExplorer(args.n_agents)
    
    buffer = ReplayBuffer(args.buffer_size, args.n_agents, args.obs_dim, STATE_DIM)
    learner = QLearner(agents, mixer, target_agents, target_mixer, optimizer, args.gamma, device)
    
    # Pętla Treningowa
    obs, state = env.reset()
    episode_reward = 0
    episodes_done = 0
    
    pbar = tqdm(total=args.total_steps, desc=f"Trening QMIX ({args.explorer})")
    
    for step in range(args.total_steps):
        eps = max(args.eps_end, args.eps_start - (args.eps_start - args.eps_end) * step / args.eps_decay)
        
        explore_mask = explorer.should_explore(eps)
        actions = []
        
        for i in range(args.n_agents):
            if explore_mask[i]:
                actions.append(env.env.action_space[0].sample())
            else:
                with torch.no_grad():
                    o_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(device)
                    q_vals = agents[i](o_tensor)
                    actions.append(q_vals.argmax(1).item())

        next_obs, next_state, rewards, done, info = env.step(actions)
        episode_reward += sum(rewards)
        
        buffer.push(obs, state, actions, rewards, next_obs, next_state, float(done))
        obs, state = next_obs, next_state
        
        if len(buffer) >= MIN_BUFFER_SIZE:
            batch = buffer.sample(args.batch_size)
            loss = learner.update(batch)
            
            if step % TARGET_UPDATE_INTERVAL == 0:
                learner.update_targets()
                
        if done:
            episodes_done += 1
            if episodes_done % 10 == 0:
                pbar.set_postfix({"Ostatnia Nagroda": episode_reward, "Epsilon": f"{eps:.2f}"})
            
            obs, state = env.reset()
            episode_reward = 0
            
        pbar.update(1)

    pbar.close()
    print("Trening zakończony!")
    
    # Zapis wag (w nazwie pliku dodajemy typ eksploratora, żeby plików nie nadpisywać!)
    weights_filename = f"agents_weights_{args.mixer}_{args.explorer}.pth"
    torch.save(agents.state_dict(), weights_filename)
    print(f"Wagi modelu zostały zapisane do pliku {weights_filename}")

if __name__ == "__main__":
    main()
