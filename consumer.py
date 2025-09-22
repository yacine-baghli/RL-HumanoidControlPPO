# fichier: consumer.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import time
from filelock import FileLock

import custom_env

# --- Configuration ---
ENV_NAME = "HumanoidObstacles-v0"
SHARED_DIR = "./shared_models/"
SAVE_PATH = "./obstacle_models/"
CONSUMER_MODEL_FILE = f"{SAVE_PATH}/obstacle_crosser_latest.zip"  # 👈 Chemin du modèle du consommateur
NUM_CPU = 8
LOGS_DIR = "./tensorboard_logs/consumer/"

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Création de l'environnement vectorisé ---
env = make_vec_env(ENV_NAME, n_envs=NUM_CPU)

# ========================================================== #
# == NOUVELLE LOGIQUE : Initialisation ou Reprise de l'entraînement == #
# ========================================================== #
model = None
# 1. On vérifie si le consommateur a déjà un modèle sauvegardé
if os.path.exists(CONSUMER_MODEL_FILE):
    print(f"Reprise de l'entraînement depuis un modèle existant : {CONSUMER_MODEL_FILE}")
    model = PPO.load(CONSUMER_MODEL_FILE, env=env, tensorboard_log=LOGS_DIR)
else:
    print("Aucun modèle 'Franchisseur' existant. En attente du premier modèle du 'Générateur'.")
    # On attend que le premier modèle du générateur soit prêt pour démarrer
    latest_info_path = f"{SHARED_DIR}/latest.txt"
    while not os.path.exists(latest_info_path):
        time.sleep(10)

    with FileLock(f"{SHARED_DIR}/latest.lock"):
        with open(latest_info_path, "r") as f:
            base_model_path = f.read()

    print(f"Démarrage avec le modèle de base : {base_model_path}")
    model = PPO.load(base_model_path, env=env, tensorboard_log=LOGS_DIR)
    model.set_timesteps(0)  # On commence le log du consommateur à zéro

# --- Boucle d'entraînement continu ---
print("[Consommateur] Début de l'entraînement continu...")
while True:
    model.learn(total_timesteps=100_000, reset_num_timesteps=False, progress_bar=True)
    # On sauvegarde la progression du consommateur après chaque session d'entraînement
    model.save(CONSUMER_MODEL_FILE)
    print(f"Progression sauvegardée dans : {CONSUMER_MODEL_FILE}")