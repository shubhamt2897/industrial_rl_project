# train_vision.py (with Checkpoints)

import gymnasium as gym
from stable_baselines3 import PPO
import os
from gymnasium.envs.registration import register

# NEW: Import the CheckpointCallback
from stable_baselines3.common.callbacks import CheckpointCallback

# --- Register the vision-based environment ---
register(
     id="FrankaReachVision-v0",
     entry_point="environments.franka_reach_env_vision:FrankaReachEnvVision",
)

# --- Parameters ---
MODEL_NAME = "PPO_Franka_Vision"
LOG_DIR = "logs"
MODELS_DIR = "trained_models"
CHECKPOINT_DIR = "checkpoints" # Folder for automatic saves
TIMESTEPS = 2_000_000 # A good number for a serious vision run

# --- RESUME CONTROL ---
# Set this to the checkpoint path if you want to resume, otherwise leave it as None
RESUME_FROM_CHECKPOINT = "checkpoints/PPO_Franka_Vision_Local_1000000_steps.zip" # or None to start new


os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Use the vision environment ---
env = gym.make("FrankaReachVision-v0", render_mode=None)

# --- NEW: Setup the Checkpoint Callback ---
# This will save a checkpoint of the model every 50,000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=50_000,
  save_path=CHECKPOINT_DIR,
  name_prefix=MODEL_NAME
)

# --- Create or Load the model ---
if RESUME_FROM_CHECKPOINT:
    print(f"Loading and resuming model from: {RESUME_FROM_CHECKPOINT}")
    model = PPO.load(
        RESUME_FROM_CHECKPOINT,
        env=env,
        tensorboard_log=LOG_DIR
    )
else:
    print("Creating a new model...")
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
    # This is crucial for resuming: it tells SB3 to continue the step count
    reset_num_timesteps=False if RESUME_FROM_CHECKPOINT else True, 
    tb_log_name=MODEL_NAME,
    callback=checkpoint_callback
)

# --- Save the final model ---
model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_final.zip")
model.save(model_path)

print(f"\n----------- Training Finished -----------")
print(f"Final model saved to: {model_path}")

env.close()