import gymnasium as gym
from stable_baselines3 import PPO
import os

# --- Configuration ---
# The name of the Gymnasium environment to use.
ENV_NAME = "Humanoid-v5"

# The filename for the saved model.
MODEL_NAME = "ppo_humanoid3"

# This is a very complex problem that requires a massive number of timesteps to learn effectively.
# Millions of steps are just a starting point to see emergent walking behavior.
TRAINING_TIMESTEPS = 1_000_000_000

# A boolean flag to switch between training a new model and loading an existing one for visualization.
# Set to True to train, False to only visualize.
DO_TRAINING = True

# --- 1. Training ---
# This block handles the training process.
# It will run if DO_TRAINING is True or if a saved model file does not already exist.
if DO_TRAINING or not os.path.exists(f"{MODEL_NAME}.zip"):
    print(f"--- Starting training for {ENV_NAME} ---")

    # Create the learning environment.
    env = gym.make(ENV_NAME)

    # Instantiate the PPO model.
    # "MlpPolicy" is a standard multi-layer perceptron policy network.
    # verbose=1 enables logging of training progress.
    # device="auto" will automatically use a GPU if available (CUDA or MPS).
    model = PPO("MlpPolicy", env, verbose=1, device="auto")

    # Start the training process. This can take a very long time!
    # A progress bar will be displayed in the console.
    model.learn(total_timesteps=TRAINING_TIMESTEPS, progress_bar=True)

    # Save the trained model to a .zip file.
    model.save(MODEL_NAME)
    print(f"--- Model saved as {MODEL_NAME}.zip ---")

    # Close the environment to free up resources.
    env.close()
else:
    print("--- Skipping training, loading existing model. ---")

# --- 2. Evaluation and Visualization ---
print("\n--- Starting visualization ---")

# Load the pre-trained model from the file.
model = PPO.load(MODEL_NAME)

# Create a new environment for visualization.
# render_mode="human" will open a window to display the simulation.
# Note: Interacting with the 3D view might be tricky at first.
env = gym.make(ENV_NAME, render_mode="human")

# Run 5 demonstration episodes.
for episode in range(5):
    # Reset the environment to get the initial observation.
    obs, info = env.reset()
    done = False
    print(f"\n--- Demonstration Episode {episode + 1} ---")

    # Loop until the episode is finished.
    while not done:
        # Get the best action from the model for the current observation.
        # deterministic=True ensures the model doesn't use random exploration.
        action, _ = model.predict(obs, deterministic=True)

        # Take the action in the environment.
        obs, reward, terminated, truncated, info = env.step(action)

        # An episode is 'done' if it's either 'terminated' (e.g., agent fell) or 'truncated' (e.g., time limit reached).
        done = terminated or truncated

# Close the visualization environment.
env.close()