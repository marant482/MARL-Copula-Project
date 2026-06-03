import argparse
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
import lbforaging

from modules.agents import MLPAgent
from envs.wrappers import SimpleEnvWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Ewaluacja wytrenowanego modelu MARL")
    
    # Parametry środowiska (muszą się zgadzać z tymi z treningu!)
    parser.add_argument("--env_id", type=str, default="Foraging-8x8-2p-2f-v3", help="ID środowiska Gym")
    parser.add_argument("--n_agents", type=int, default=2, help="Liczba agentów")
    parser.add_argument("--n_actions", type=int, default=6, help="Liczba dostępnych akcji")
    parser.add_argument("--obs_dim", type=int, default=12, help="Wymiar wektora obserwacji")
    
    # Parametry ewaluacji
    parser.add_argument("--explorer", type=str, default="copula", choices=["copula", "epsilon"], help="Które wagi wczytać?")
    parser.add_argument("--max_steps", type=int, default=50, help="Maksymalna liczba kroków animacji")
    
    return parser.parse_args()

def evaluate():
    args = parse_args()
    GRID_SIZE = 8 # Specyficzne dla domyślnego Level-Based Foraging
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rozpoczynam ewaluację na: {device}")
    print(f"Środowisko: {args.env_id} | Model z eksploracji: {args.explorer}")

    # Inicjalizacja
    raw_env = gym.make(args.env_id)
    env = SimpleEnvWrapper(raw_env, args.n_agents, args.n_actions)

    # Ładowanie wytrenowanych agentów
    agents = nn.ModuleList([MLPAgent(args.obs_dim, args.n_actions).to(device) for _ in range(args.n_agents)])
    weights_filename = f"agents_weights_{args.explorer}.pth"
    
    try:
        agents.load_state_dict(torch.load(weights_filename, map_location=device, weights_only=True))
        print(f"Pomyślnie wczytano wagi z pliku {weights_filename}")
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku {weights_filename}! Najpierw odpal trening dla tego eksperymentu.")
        return
        
    agents.eval() # Wyłącza tryb treningowy (np. dropout)

    obs, _ = env.reset()
    frames = []
    done = False
    step = 0

    print("Renderowanie animacji...")

    while not done and step < args.max_steps:
        # Rysowanie planszy (Kod specyficzny dla Level-Based Foraging!)
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

        # Wybór akcji (100% zachłannie, zero losowania)
        actions = []
        for i in range(args.n_agents):
            with torch.no_grad():
                o_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(device)
                q_vals = agents[i](o_tensor)
                actions.append(q_vals.argmax(1).item())

        next_obs, _, _, done, _ = env.step(actions)
        obs = next_obs
        step += 1

    # Zapis GIFa
    gif_path = f"eval_{args.explorer}.gif"
    if frames:
        imageio.mimsave(gif_path, frames, fps=5)
        print(f"Ewaluacja zakończona! Zapisano animację jako: {gif_path}")
    else:
        print("Brak klatek do zapisania (czy na pewno używasz LBF?).")

if __name__ == "__main__":
    evaluate()
