import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import copy
import numpy as np
from tqdm import tqdm
import wandb

from modules.agents import MLPAgent, RNNAgent
from modules.mixers import QMixMixer, VDNMixer
from modules.explorers import GaussianCopulaExplorer, EpsilonGreedyExplorer, ActionCopulaSampler
from envs.wrappers import SimpleEnvWrapper
from utils.replay_buffer import ReplayBuffer, EpisodicReplayBuffer
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
    parser.add_argument("--use_bptt", action="store_true",
                        help="Używaj Episodic Buffer i Backpropagation Through Time dla sieci RNN")
    parser.add_argument("--max_steps", type=int, default=50, help="Maksymalna długość epizodu w Episodic Buffer")
    
    # Parametry eksploracji
    parser.add_argument("--explorer", type=str, default="copula", choices=["copula", "epsilon"])
    parser.add_argument("--copula_corr", type=float, default=0.7)
    parser.add_argument("--action_copula_corr", type=float, default=0.0,
                        help="Korelacja dla losowanych akcji (0.0 = niezależne losowanie, 1.0 = identyczne akcje)")
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=int, default=20000)
    
    # Parametry ewaluacji
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--eval_episodes", type=int, default=10)

    # Parametry stabilności RL
    parser.add_argument("--independent_agents", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--target_update", type=int, default=5000)
    parser.add_argument("--reward_priority", type=float, default=0.25,
                        help="Odsetek batcha (0.0 do 1.0) rezerwowany dla kroków/epizodów z nagrodą. Ustawienie na 0.0 całkowicie wyłącza ten mechanizm.")

    # Parametry architektury sieci
    parser.add_argument("--agent_type", type=str, default="rnn", choices=["rnn", "mlp"], help="Typ sieci agenta (domyślnie rnn)")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    
    return parser.parse_args()

def evaluate_model(env_id, n_agents, n_actions, agents, device, eval_episodes, agent_type, hidden_dim):
    eval_raw_env = gym.make(env_id)
    eval_env = SimpleEnvWrapper(eval_raw_env, n_agents, n_actions)
    total_rewards = []
    
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0
        
        if agent_type == "rnn":
            hiddens = torch.zeros(n_agents, hidden_dim).to(device)
        
        while not done:
            actions = []
            for i in range(n_agents):
                with torch.no_grad():
                    o_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(device)
                    if agent_type == "rnn":
                        q_vals, h_next = agents[i](o_tensor, hiddens[i].unsqueeze(0))
                        hiddens[i] = h_next.squeeze(0)
                    else:
                        q_vals = agents[i](o_tensor)
                    actions.append(q_vals.argmax(1).item())
                    
            next_obs, _, rewards, terminated, truncated, _ = eval_env.step(actions)
            done = bool(np.any(terminated) or np.any(truncated))
            ep_reward += sum(rewards)
            obs = next_obs
            
        total_rewards.append(ep_reward)
    return np.mean(total_rewards)

def main():
    args = parse_args()
    
    wandb.init(
        project="MARL-Copula-Project",
        name=f"{args.env_id}_{args.mixer}_{args.explorer}_{args.agent_type}_rho{args.copula_corr}",
        config=vars(args)
    )
    
    STATE_DIM = args.obs_dim * args.n_agents
    TARGET_UPDATE_INTERVAL = args.target_update
    MIN_BUFFER_SIZE = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_env = gym.make(args.env_id)
    env = SimpleEnvWrapper(raw_env, args.n_agents, args.n_actions)
    
    # Inicjalizacja sieci w zależności od agent_type
    if args.agent_type == "rnn":
        if args.independent_agents:
            agents = nn.ModuleList([RNNAgent(args.obs_dim, args.n_actions, hidden_dim=args.hidden_dim).to(device) for _ in range(args.n_agents)])
        else:
            shared_agent = RNNAgent(args.obs_dim, args.n_actions, hidden_dim=args.hidden_dim).to(device)
            agents = nn.ModuleList([shared_agent for _ in range(args.n_agents)])
    else:
        if args.independent_agents:
            agents = nn.ModuleList([MLPAgent(args.obs_dim, args.n_actions, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device) for _ in range(args.n_agents)])
        else:
            shared_agent = MLPAgent(args.obs_dim, args.n_actions, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
            agents = nn.ModuleList([shared_agent for _ in range(args.n_agents)])
        
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

    action_sampler = ActionCopulaSampler(args.n_agents, args.n_actions, correlation=args.action_copula_corr)

    if args.use_bptt:
        ep_capacity = max(1, args.buffer_size // args.max_steps)
        buffer = EpisodicReplayBuffer(ep_capacity, args.max_steps, args.n_agents, args.obs_dim, STATE_DIM,
                                      reward_priority=args.reward_priority)
        min_buffer_size = max(1, MIN_BUFFER_SIZE // args.max_steps)
    else:
        buffer = ReplayBuffer(args.buffer_size, args.n_agents, args.obs_dim, STATE_DIM, hidden_dim=args.hidden_dim,
                              reward_priority=args.reward_priority)
        min_buffer_size = MIN_BUFFER_SIZE
    learner = QLearner(agents, mixer, target_agents, target_mixer, optimizer, args.gamma, device, grad_clip=args.grad_clip)
    
    obs, state = env.reset()
    episode_reward = 0
    episodes_done = 0
    
    # Inicjalizacja stanów ukrytych dla pierwszego epizodu
    hiddens = torch.zeros(1, args.n_agents, args.hidden_dim).to(device) if args.agent_type == "rnn" else None

    if args.use_bptt:
        episode_data = {'obs': [], 'states': [], 'actions': [], 'rewards': [], 'next_obs': [], 'next_states': [],
                        'dones': []}

    pbar = tqdm(total=args.total_steps, desc=f"Trening {args.mixer.upper()} ({args.explorer})")

    for step in range(1, args.total_steps + 1):
        eps = max(args.eps_end, args.eps_start - (args.eps_start - args.eps_end) * step / args.eps_decay)
        explore_mask = explorer.should_explore(eps)
        random_actions = action_sampler.sample()
        actions = []

        next_hiddens = torch.zeros(1, args.n_agents, args.hidden_dim).to(device) if args.agent_type == "rnn" else None

        if not args.independent_agents:
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).to(device)
            with torch.no_grad():
                if args.agent_type == "rnn":
                    q_vals, next_hiddens = agents[0](obs_tensor, hiddens)
                else:
                    q_vals = agents[0](obs_tensor)
                greedy_actions = q_vals.argmax(dim=1).cpu().numpy()

            for i in range(args.n_agents):
                if explore_mask[i]:
                    actions.append(random_actions[i])
                else:
                    actions.append(int(greedy_actions[i]))
        else:
            for i in range(args.n_agents):
                if explore_mask[i]:
                    actions.append(random_actions[i])
                else:
                    with torch.no_grad():
                        o_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(device)
                        if args.agent_type == "rnn":
                            # Wycięcie stanu ukrytego tylko dla konkretnego agenta
                            q_vals, h_next = agents[i](o_tensor, hiddens[:, i:i + 1, :])
                            next_hiddens[:, i:i + 1, :] = h_next
                        else:
                            q_vals = agents[i](o_tensor)
                        actions.append(q_vals.argmax(1).item())

        next_obs, next_state, rewards, terminated, truncated, info = env.step(actions)
        episode_reward += sum(rewards)
        true_done = float(np.any(terminated))

        # Zapisujemy do bufora
        loss_val = None

        if args.use_bptt:
            episode_data['obs'].append(obs)
            episode_data['states'].append(state)
            episode_data['actions'].append(actions)
            global_reward = sum(rewards) if isinstance(rewards, (list, np.ndarray)) else rewards
            episode_data['rewards'].append([global_reward])
            episode_data['next_obs'].append(next_obs)
            episode_data['next_states'].append(next_state)
            episode_data['dones'].append([true_done])

            if np.any(terminated) or np.any(truncated):
                buffer.push_episode(episode_data)
                episode_data = {'obs': [], 'states': [], 'actions': [], 'rewards': [], 'next_obs': [],
                                'next_states': [], 'dones': []}

                # === KLUCZOWA ZMIANA: Trenujemy tylko po zapisaniu epizodu! ===
                if len(buffer) >= min_buffer_size:
                    current_batch_size = min(args.batch_size, len(buffer))
                    batch = buffer.sample(current_batch_size)
                    loss_val = learner.update_bptt(batch)

                    if step % TARGET_UPDATE_INTERVAL == 0:
                        learner.update_targets()

            if args.agent_type == "rnn":
                hiddens = next_hiddens
        else:
            # Tryb MLP krok-po-kroku
            if args.agent_type == "rnn":
                buffer.push(obs, state, actions, rewards, next_obs, next_state, true_done,
                            hiddens=hiddens.squeeze(0).cpu().numpy(),
                            next_hiddens=next_hiddens.squeeze(0).cpu().numpy())
                hiddens = next_hiddens
            else:
                buffer.push(obs, state, actions, rewards, next_obs, next_state, true_done)

            # Trening MLP co każdy krok gry
            if len(buffer) >= min_buffer_size:
                batch = buffer.sample(args.batch_size)
                loss_val = learner.update(batch)

                if step % TARGET_UPDATE_INTERVAL == 0:
                    learner.update_targets()

        obs, state = next_obs, next_state

        # Zapis logów do WandB na koniec epizodu
        if np.any(terminated) or np.any(truncated):
            episodes_done += 1
            metrics_to_log = {
                "Train/Episode_Reward": episode_reward,
                "Train/Epsilon": eps,
                "Train/Episodes_Total": episodes_done
            }
            if loss_val is not None:
                metrics_to_log["Train/Loss"] = loss_val

            wandb.log(metrics_to_log, step=step)

            obs, state = env.reset()
            episode_reward = 0
            if args.agent_type == "rnn":
                hiddens = torch.zeros(1, args.n_agents, args.hidden_dim).to(device)
            
        if step % args.eval_interval == 0:
            mean_eval_reward = evaluate_model(args.env_id, args.n_agents, args.n_actions, agents, device, args.eval_episodes, args.agent_type, args.hidden_dim)
            # FIX: Przekazujemy step=step do wandb.log
            wandb.log({"Eval/Mean_Reward": mean_eval_reward}, step=step)
            pbar.set_postfix({"Eval Reward": f"{mean_eval_reward:.2f}", "Epsilon": f"{eps:.2f}"})
            
        pbar.update(1)

    pbar.close()
    weights_filename = f"agents_weights_{args.mixer}_{args.explorer}_{args.agent_type}.pth"
    torch.save(agents.state_dict(), weights_filename)
    wandb.finish()

if __name__ == "__main__":
    main()
