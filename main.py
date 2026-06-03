import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import copy
import numpy as np
from tqdm import tqdm
import wandb

from modules.agents import MLPAgent
from modules.mixers import QMixMixer, VDNMixer
from modules.explorers import GaussianCopulaExplorer, EpsilonGreedyExplorer
from envs.wrappers import SimpleEnvWrapper
from utils.replay_buffer import ReplayBuffer
from learners.q_learner import QLearner

import lbforaging

def parse_args():
    parser = argparse.ArgumentParser(description="Trening MARL z Kopulą Gaussa")
    
    # Parametry środowiska
    parser.add_argument("--env_id", type=str, default="Foraging-8x8-2p-2f-v3")
    parser.add_argument("--n_agents", type=int, default=2)
    parser.add_argument("--n_actions", type=int, default=6)
    parser.add_argument("--obs_dim", type=int, default=12)
    
    # Parametry treningu
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--mixer", type=str, default="qmix", choices=["qmix", "vdn"])
    
    # Parametry eksploracji
    parser.add_argument("--explorer", type=str, default="copula", choices=["copula", "epsilon"])
    parser.add_argument("--copula_corr", type=float, default=0.7)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=int, default=20000)
    
    # Parametry ewaluacji (NOWE)
    parser.add_argument("--eval_interval", type=int, default=5000, help="Co ile kroków testować model")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Liczba epizodów testowych")

    # Parametry stabilności RL
    parser.add_argument("--independent_agents", action="store_true", help="Jeśli flaga jest podana, agenci mają oddzielne sieci (domyślnie: współdzielą jedną)")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Maksymalna norma gradientu (clipping)")
    parser.add_argument("--target_update", type=int, default=5000, help="Co ile kroków aktualizować sieć docelową")

    # Parametry architektury sieci
    parser.add_argument("--hidden_dim", type=int, default=128, help="Liczba neuronów w warstwie")
    parser.add_argument("--num_layers", type=int, default=2, help="Liczba warstw ukrytych")
    
    return parser.parse_args()

def evaluate_model(env_id, n_agents, n_actions, agents, device, eval_episodes):
    """Przeprowadza czystą ewaluację bez eksploracji i zwraca średnią nagrodę."""
    eval_raw_env = gym.make(env_id)
    eval_env = SimpleEnvWrapper(eval_raw_env, n_agents, n_actions)
    
    total_rewards = []
    
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            actions = []
            for i in range(n_agents):
                with torch.no_grad():
                    o_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(device)
                    q_vals = agents[i](o_tensor)
                    actions.append(q_vals.argmax(1).item())
                    
            next_obs, _, rewards, terminated, truncated, _ = eval_env.step(actions)
            
            # W ewaluacji koniec epizodu następuje przy którymkolwiek z tych warunków
            done = bool(np.any(terminated) or np.any(truncated))
            
            ep_reward += sum(rewards)
            obs = next_obs
            
        total_rewards.append(ep_reward)
        
    return np.mean(total_rewards)

def main():
    args = parse_args()
    
    # 1. Inicjalizacja Weights & Biases
    wandb.init(
        project="MARL-Copula-Project",
        name=f"{args.env_id}_{args.mixer}_{args.explorer}_rho{args.copula_corr}",
        config=vars(args)
    )
    
    STATE_DIM = args.obs_dim * args.n_agents
    TARGET_UPDATE_INTERVAL = args.target_update
    MIN_BUFFER_SIZE = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_env = gym.make(args.env_id)
    env = SimpleEnvWrapper(raw_env, args.n_agents, args.n_actions)
    
    #agents = nn.ModuleList([MLPAgent(args.obs_dim, args.n_actions).to(device) for _ in range(args.n_agents)])
    
    # Inicjalizacja sieci agentów (Współdzielenie wag lub oddzielne sieci)
    if args.independent_agents:
        agents = nn.ModuleList([MLPAgent(args.obs_dim, args.n_actions, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device) for _ in range(args.n_agents)])
    else:
        shared_agent = MLPAgent(args.obs_dim, args.n_actions, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
        agents = nn.ModuleList([shared_agent for _ in range(args.n_agents)]) # Wszyscy wskazują na jeden "mózg"
        
    
    if args.mixer == "qmix":
        mixer = QMixMixer(args.n_agents, STATE_DIM).to(device)
    elif args.mixer == "vdn":
        mixer = VDNMixer().to(device)
        
    target_agents = copy.deepcopy(agents)
    target_mixer = copy.deepcopy(mixer)
    
    optimizer = optim.Adam(list(agents.parameters()) + list(mixer.parameters()), lr=args.lr)
    
    if args.explorer == "copula":
        explorer = GaussianCopulaExplorer(args.n_agents, correlation=args.copula_corr)
    else:
        explorer = EpsilonGreedyExplorer(args.n_agents)
    
    buffer = ReplayBuffer(args.buffer_size, args.n_agents, args.obs_dim, STATE_DIM)
    # Podmieniamy tę linijkę:
    learner = QLearner(agents, mixer, target_agents, target_mixer, optimizer, args.gamma, device, grad_clip=args.grad_clip)
    
    obs, state = env.reset()
    episode_reward = 0
    episodes_done = 0
    
    pbar = tqdm(total=args.total_steps, desc=f"Trening {args.mixer.upper()} ({args.explorer})")
    
    for step in range(1, args.total_steps + 1):
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

        next_obs, next_state, rewards, terminated, truncated, info = env.step(actions)
        episode_reward += sum(rewards)

        true_done = float(np.any(terminated))
        buffer.push(obs, state, actions, rewards, next_obs, next_state, true_done)
        obs, state = next_obs, next_state
        done = bool(np.any(terminated) or np.any(truncated))
        
        loss_val = None
        if len(buffer) >= MIN_BUFFER_SIZE:
            batch = buffer.sample(args.batch_size)
            loss_val = learner.update(batch)
            
            if step % TARGET_UPDATE_INTERVAL == 0:
                learner.update_targets()
                
        if np.any(terminated) or np.any(truncated):
            episodes_done += 1
            
            # 2. Logowanie metryk treningowych na bieżąco
            metrics_to_log = {
                "Train/Episode_Reward": episode_reward,
                "Train/Epsilon": eps,
                "Global_Step": step
            }
            if loss_val is not None:
                metrics_to_log["Train/Loss"] = loss_val
                
            wandb.log(metrics_to_log)
            
            obs, state = env.reset()
            episode_reward = 0
            
        # 3. Okresowa Ewaluacja (Testowanie modelu)
        if step % args.eval_interval == 0:
            mean_eval_reward = evaluate_model(args.env_id, args.n_agents, args.n_actions, agents, device, args.eval_episodes)
            wandb.log({
                "Eval/Mean_Reward": mean_eval_reward,
                "Global_Step": step
            })
            pbar.set_postfix({"Eval Reward": f"{mean_eval_reward:.2f}", "Epsilon": f"{eps:.2f}"})
            
        pbar.update(1)

    pbar.close()
    
    weights_filename = f"agents_weights_{args.mixer}_{args.explorer}.pth"
    torch.save(agents.state_dict(), weights_filename)
    
    # 4. Zakończenie pracy z WandB
    wandb.finish()

if __name__ == "__main__":
    main()
