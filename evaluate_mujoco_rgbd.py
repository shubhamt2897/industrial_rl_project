# evaluate_mujoco_rgbd.py

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import os
from gymnasium.envs.registration import register
# NEW: Import the RecordVideo wrapper
from gymnasium.wrappers import RecordVideo

# --- Register the MUJOCO environment ---
register(
     id="FrankaReachMujocoRGBD-v0",
     entry_point="environments.franka_reach_env_mujoco:FrankaReachEnvMujoco",
)

# --- Parameters ---
MODEL_NAME = "PPO_Franka_Mujoco_RGBD"
MODEL_PATH = f"trained_models_mujoco/{MODEL_NAME}.zip"
RECORDINGS_DIR = "mujoco_results"
EPISODES_TO_RUN = 20

# --- Load the environment ---
# Note: We don't set render_mode here because the wrapper will handle it
env = gym.make("FrankaReachMujocoRGBD-v0")

# --- NEW: Wrap the environment with the RecordVideo wrapper ---
# This will automatically save a video for each episode in the specified folder
env = RecordVideo(env, video_folder=RECORDINGS_DIR, name_prefix=f"{MODEL_NAME}-eval")

# --- Load the trained model ---
model = PPO.load(MODEL_PATH, env=env)

# --- Evaluation Metrics ---
successful_episodes = 0
steps_per_episode = []

# --- Run the evaluation ---
for ep in range(EPISODES_TO_RUN):
    obs, info = env.reset()
    done = False
    episode_steps = 0
    print(f"\n--- Starting Episode {ep+1}/{EPISODES_TO_RUN} ---")

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_steps += 1
        
        if terminated or truncated:
            done = True
            if terminated:
                print(f"  SUCCESS! Reached target in {episode_steps} steps.")
                successful_episodes += 1
                steps_per_episode.append(episode_steps)
            else:
                print(f"  FAILURE! Timed out after {episode_steps} steps.")

# --- Final Report ---
print("\n\n--- Evaluation Finished ---")
print(f"Recordings saved to: {RECORDINGS_DIR}")
success_rate = (successful_episodes / EPISODES_TO_RUN) * 100
print(f"Success Rate: {success_rate:.2f}% ({successful_episodes}/{EPISODES_TO_RUN})")

if successful_episodes > 0:
    avg_steps = np.mean(steps_per_episode)
    print(f"Average Steps per Successful Episode: {avg_steps:.2f}")

env.close()