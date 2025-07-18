# train.py

import gymnasium as gym
from stable_baselines3 import PPO
import os
from gymnasium.envs.registration import register

# Register the custom environment
register(
     id="FrankaReach-v0",
     entry_point="environments.franka_reach_env:FrankaReachEnv",
)

# Create directories to save models and logs
models_dir = "trained_models"
logdir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Instantiate the environment in 'human' mode to watch
env = gym.make("FrankaReach-v0", render_mode='none')
env.reset()

# Define the RL model (PPO)
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=logdir,
    # Adding a device argument can prevent the GPU warning
     device="cpu" ,
    learning_rate=0.0001,
    ent_coef=0.01 # <--- Added entropy coefficient for exploration model 4
)

# Train the model
TIMESTEPS = 5_000_000 #Increased for model 3 till model 2 it was 1_000_000, for model 5 increased to 5000000
model.learn(
    total_timesteps=TIMESTEPS, 
    reset_num_timesteps=False, 
    tb_log_name="PPO_Franka_Reach_m4"  # Updated for model 4
)

# Save the trained model
model_path = os.path.join(models_dir, "ppo_franka_reach_model5.zip")
model.save(model_path)

print(f"\n----------- Training Finished -----------")
print(f"Model saved to: {model_path}")

# Clean up
env.close()