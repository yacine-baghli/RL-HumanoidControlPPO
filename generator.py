# fichier: generator.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import time
from filelock import FileLock

# --- Configuration ---
ENV_NAME = "Humanoid-v5"
SHARED_DIR = "./shared_models/"
SAVE_INTERVAL = 400_000
NUM_CPU = 8
LOGS_DIR = "./tensorboard_logs/generator/"  # 👈 Dossier de log pour le générateur

os.makedirs(SHARED_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Création de l'environnement vectorisé ---
env = make_vec_env(ENV_NAME, n_envs=NUM_CPU)

# --- Configuration du modèle PPO ---
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
    n_steps=2048,
    batch_size=64,
    tensorboard_log=LOGS_DIR  # 👈 Ajout de l'argument TensorBoard
)

print("[Générateur] Début de l'entraînement parallèle...")

version = 0
while True:
    model.learn(total_timesteps=SAVE_INTERVAL, reset_num_timesteps=False, progress_bar=True)

    version += 1
    model_path = f"{SHARED_DIR}/walker_v{version}.zip"
    latest_info_path = f"{SHARED_DIR}/latest.txt"
    lock_path = f"{SHARED_DIR}/latest.lock"

    lock = FileLock(lock_path)

    print(f"[Générateur] Sauvegarde du modèle v{version}...")
    with lock:
        model.save(model_path)
        with open(latest_info_path, "w") as f:
            f.write(model_path)

    print(f"[Générateur] Modèle v{version} sauvegardé et publié.")