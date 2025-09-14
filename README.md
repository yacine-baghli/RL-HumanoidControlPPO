## Humanoid Control using Deep Reinforcement Learning with PPO

This repository contains an implementation for training a 3D humanoid agent to walk using Deep Reinforcement Learning. The project leverages the Humanoid-v5 environment from the Gymnasium library and employs the Proximal Policy Optimization (PPO) algorithm from Stable-Baselines3, a robust library for RL research and development.

The core objective is to solve a complex, high-dimensional continuous control problem, demonstrating the effectiveness of PPO in learning sophisticated locomotion behaviors.

### 🚀 Core Features
End-to-End Workflow: Provides a complete pipeline from model training and serialization to inference and visualization.

High-Performance Training: Utilizes Stable-Baselines3 for efficient, state-of-the-art PPO implementation, with support for GPU acceleration (device="auto").

Configurable Architecture: Key parameters such as the environment, model identifiers, and training duration are modularized for easy experimentation.

Inference & Visualization: Includes a mode for loading a pre-trained policy and rendering its performance in the simulated environment.

### 🛠️ Prerequisites & Installation
To run this project, a Python environment with the necessary libraries is required. Using a virtual environment is strongly recommended to manage dependencies.

Clone the Repository:

```bash

git clone https://github.com/yacine-baghli/RL-HumanoidControlPPO
cd RL-HumanoidControlPPO
```

Set up a Virtual Environment:

```bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install Dependencies:
The implementation relies on PyTorch for the backend.

```bash
pip install gymnasium stable-baselines3[extra] torch
```
Note: stable-baselines3[extra] includes additional dependencies that can be useful for logging and advanced callbacks.

### ⚙️ Execution & Usage
The script humanoid.py is designed for two primary modes of operation, controlled by the DO_TRAINING boolean flag.

1. Training a New Model
To initiate a training session from scratch:

In humanoid.py, set the DO_TRAINING flag to True.

```Python
DO_TRAINING = True
```
Adjust TRAINING_TIMESTEPS as needed. The Humanoid-v5 environment is notoriously difficult and requires a substantial number of timesteps (typically >10 million) to converge to a stable walking gait. The current value is set high for a comprehensive run.

Execute the script from your terminal:

```Bash
python humanoid.py
```
The training process will begin, logging output to the console. Upon completion, the trained policy will be serialized to a .zip file specified by the MODEL_NAME variable (e.g., ppo_humanoid3.zip).

2. Visualizing a Pre-Trained Model
To run inference and visualize the agent's performance with an existing model:

Ensure the trained model file (e.g., ppo_humanoid3.zip) is present in the root directory.

Set the DO_TRAINING flag to False.

```Python

DO_TRAINING = False
```
Execute the script:

```Bash

python humanoid.py
```
The script will load the serialized model and launch the Gymnasium environment with render_mode="human", running several episodes to demonstrate the learned policy.

### 🔧 Configuration Parameters
The script's behavior can be easily modified via the following global variables:

ENV_NAME: The target Gymnasium environment ID. Default: "Humanoid-v5".

MODEL_NAME: The filename for saving and loading the trained model.

TRAINING_TIMESTEPS: The total number of environment steps for the training phase.

DO_TRAINING: A boolean flag to switch between training and visualization modes.

### 🤖 About the Humanoid-v5 Environment
The Humanoid-v5 environment presents a significant challenge in reinforcement learning. The goal is to train a simulated humanoid to walk forward as far as possible without falling.

State Space: High-dimensional, including joint angles, velocities, and center-of-mass coordinates.

Action Space: Continuous, corresponding to the torque applied to each of the humanoid's joints.

Reward Function: Rewards are given for forward velocity and staying alive, with penalties for high energy consumption or joint limits.

Successfully solving this environment demonstrates a robust understanding of deep reinforcement learning for continuous control tasks.
