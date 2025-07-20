# train_vision.py (Corrected Resume Logic)

import gymnasium as gym
from stable_baselines3 import PPO
import os
import re
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import CheckpointCallback

# --- Register the vision-based environment ---
register(
     id="FrankaReachVision-v0",
     entry_point="environments.franka_reach_env_vision:FrankaReachEnvVision",
)

# --- Parameters ---
# CORRECTED: This name must EXACTLY match the prefix of your saved checkpoint files
MODEL_NAME = "PPO_Franka_Vision"
LOG_DIR = "logs"
MODELS_DIR = "trained_models"
CHECKPOINT_DIR = "checkpoints"
TOTAL_TIMESTEPS = 2_000_000 

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Setup Environment ---
env = gym.make("FrankaReachVision-v0", render_mode=None)

# --- Setup Checkpoint Callback ---
checkpoint_callback = CheckpointCallback(
  save_freq=50_000,
  save_path=CHECKPOINT_DIR,
  name_prefix=MODEL_NAME
)

# --- Smart Resume Logic ---
latest_checkpoint = None
if os.path.exists(CHECKPOINT_DIR) and len(os.listdir(CHECKPOINT_DIR)) > 0:
    # Find all checkpoint files that match the model name
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith(MODEL_NAME) and f.endswith(".zip")]
    if checkpoints:
        # Sort by the step number in the filename to find the latest
        checkpoints.sort(key=lambda f: int(re.search(r'_(\d+)_steps', f).group(1)))
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
        print(f"Found latest checkpoint: {latest_checkpoint}")

# --- Create or Load the model ---
if latest_checkpoint:
    print(f"Loading and resuming model from: {latest_checkpoint}")
    # When loading, SB3 automatically handles setting the device
    model = PPO.load(
        latest_checkpoint,
        env=env,
        tensorboard_log=LOG_DIR
    )
else:
    print("No checkpoints found. Creating a new model...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cuda",
        learning_rate=0.0001,
        n_steps=1024,
        batch_size=64,
        ent_coef=0.01,
    )

# --- Continue or Start Training ---
print("Starting training...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    # Crucial for resuming: continue step count if loading, reset if new
    reset_num_timesteps=False if latest_checkpoint else True, 
    tb_log_name=MODEL_NAME,
    callback=checkpoint_callback
)

# --- Save the final model ---
model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_final.zip")
model.save(model_path)

print(f"\n----------- Training Finished -----------")
print(f"Final model saved to: {model_path}")

env.close()