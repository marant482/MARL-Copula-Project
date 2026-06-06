import argparse
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
import lbforaging

from modules.agents import MLPAgent, RNNAgent
from envs.wrappers import SimpleEnvWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Ewaluacja wytrenowanego modelu MARL")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--env_id", type=str, default="Foraging-8x8-2p-2f-v3")
    parser.add_argument("--mixer", type=str, default="qmix", choices=["qmix", "vdn"])
    parser.add_argument("--n_agents", type=int, default=2)
    parser.add_argument("--n_actions", type=int, default=6)
    parser.add_argument("--obs_dim", type=int, default=12)
    parser.add_argument("--explorer", type=str, default="copula", choices=["copula", "epsilon"])
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--independent_agents", action="store_true")
    parser.add_argument("--agent_type", type=str, default="rnn", choices=["rnn", "mlp"])
    return parser.parse_args()

def evaluate():
    args = parse_args()
    GRID_SIZE = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    raw_env = gym.make(args.env_id)
    env = SimpleEnvWrapper(raw_env, args.n_agents, args.n_actions)

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

    weights_filename = f"agents_weights_{args.mixer}_{args.explorer}_{args.agent_type}.pth"
    
    try:
        agents.load_state_dict(torch.load(weights_filename, map_location=device, weights_only=True))
        print(f"Pomyślnie wczytano wagi z pliku {weights_filename}")
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku {weights_filename}!")
        return
        
    agents.eval()
    obs, _ = env.reset()
    frames = []
    done = False
    step = 0

    if args.agent_type == "rnn":
        hiddens = torch.zeros(args.n_agents, args.hidden_dim).to(device)

    print("Renderowanie animacji...")
    while not done and step < args.max_steps:
        if "Foraging" in args.env_id:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(-0.5, GRID_SIZE - 0.5)
            ax.set_ylim(-0.5, GRID_SIZE - 0.5)
            ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1))
            ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1))
            ax.grid()
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            for i, player in enumerate(raw_env.unwrapped.players):
                if player.position is not None:
                    r, c = player.position
                    ax.scatter(c, GRID_SIZE - 1 - r, s=500, label=f"Agent {i+1}")

            field = raw_env.unwrapped.field
            food_positions = np.argwhere(field > 0)
            for r, c in food_positions:
                ax.text(c, GRID_SIZE - 1 - r, "A", fontsize=24, ha="center", va="center")

            ax.set_title(f"{args.env_id} — krok {step}")
            ax.legend(loc="upper right")
            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(image)
            plt.close(fig)

        actions = []
        for i in range(args.n_agents):
            with torch.no_grad():
                o_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(device)
                if args.agent_type == "rnn":
                    q_vals, h_next = agents[i](o_tensor, hiddens[i].unsqueeze(0))
                    hiddens[i] = h_next.squeeze(0)
                else:
                    q_vals = agents[i](o_tensor)
                actions.append(q_vals.argmax(1).item())

        next_obs, _, _, terminated, truncated, _ = env.step(actions)
        done = bool(np.any(terminated) or np.any(truncated))
        step += 1

    gif_path = f"eval_{args.explorer}_{args.agent_type}.gif"
    if frames:
        imageio.mimsave(gif_path, frames, fps=5)
        print(f"Zapisano animację jako: {gif_path}")

if __name__ == "__main__":
    evaluate()
