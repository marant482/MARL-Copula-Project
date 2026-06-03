import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
import lbforaging

from modules.agents import MLPAgent
from envs.wrappers import SimpleEnvWrapper

def evaluate():
    # Konfiguracja (taka sama jak w main.py)
    ENV_ID = "Foraging-8x8-2p-2f-v3"
    N_AGENTS = 2
    N_ACTIONS = 6
    OBS_DIM = 12
    GRID_SIZE = 8
    MAX_STEPS = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicjalizacja
    raw_env = gym.make(ENV_ID)
    env = SimpleEnvWrapper(raw_env, N_AGENTS, N_ACTIONS)

    # Ładowanie wytrenowanych agentów
    agents = nn.ModuleList([MLPAgent(OBS_DIM, N_ACTIONS).to(device) for _ in range(N_AGENTS)])
    agents.load_state_dict(torch.load("agents_weights.pth", map_location=device, weights_only=True))
    agents.eval() # Ustawienie sieci w tryb ewaluacji (wyłącza np. dropout jeśli by był)

    obs, _ = env.reset()
    frames = []
    done = False
    step = 0

    print("Rozpoczynam ewaluację i renderowanie...")

    while not done and step < MAX_STEPS:
        # Rysowanie planszy (z Twojego oryginalnego kodu)
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
            ax.text(c, GRID_SIZE - 1 - r, "A", fontsize=24, ha="center", va="center") # "A" zamiast emoji dla uniknięcia błędów czcionki

        ax.set_title(f"LBForaging — krok {step}")
        ax.legend(loc="upper right")

        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(image)
        plt.close(fig)

        # Wybór akcji (Tylko najlepsze akcje - zero losowości)
        actions = []
        for i in range(N_AGENTS):
            with torch.no_grad():
                o_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(device)
                q_vals = agents[i](o_tensor)
                actions.append(q_vals.argmax(1).item())

        next_obs, _, _, done, _ = env.step(actions)
        obs = next_obs
        step += 1

    # Zapis GIFa
    gif_path = "eval_render.gif"
    imageio.mimsave(gif_path, frames, fps=5)
    print(f"Ewaluacja zakończona! Zapisano animację jako: {gif_path}")

if __name__ == "__main__":
    evaluate()
