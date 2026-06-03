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
    weights_filename = f"agents
