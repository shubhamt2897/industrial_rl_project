# evaluate_vision.py

import gymnasium as gym
import time
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
import os

# Import and register our custom VISION environment
from gymnasium.envs.registration import register
register(id="FrankaReachVision-v0", entry_point="environments.franka_reach_env_vision:FrankaReachEnvVision")

# --- Parameters ---
# --- CORRECTED MODEL PATH ---
# This now exactly matches the filename in your checkpoints folder
MODEL_PATH = "checkpoints/PPO_Franka_Vision_1450000_steps" 

RECORDINGS_DIR = "vision_model_1M_results"
EPISODES_TO_RUN = 20

os.makedirs(RECORDINGS_DIR, exist_ok=True)

# --- Load the environment and the trained model ---
env = gym.make("FrankaReachVision-v0", render_mode='human')
model = PPO.load(MODEL_PATH, env=env)

# --- Evaluation Metrics ---
successful_episodes = 0
steps_per_episode = []

# --- Run the evaluation ---
for ep in range(EPISODES_TO_RUN):
    obs, info = env.reset()
    
    video_filename = os.path.join(RECORDINGS_DIR, f"episode_{ep+1}.mp4")
    p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_filename)
    
    done = False
    episode_steps = 0
    print(f"\n--- Starting Episode {ep+1}/{EPISODES_TO_RUN} ---")

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_steps += 1
        
        if terminated or truncated:
            done = True
            p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
            if terminated:
                print(f"  SUCCESS! Reached target in {episode_steps} steps.")
                successful_episodes += 1
                steps_per_episode.append(episode_steps)
            else:
                print(f"  FAILURE! Timed out after {episode_steps} steps.")
            print(f"  - Recording saved to {video_filename}")

# --- Final Report ---
print("\n\n--- Evaluation Finished ---")
success_rate = (successful_episodes / EPISODES_TO_RUN) * 100
print(f"Success Rate: {success_rate:.2f}% ({successful_episodes}/{EPISODES_TO_RUN})")

if successful_episodes > 0:
    avg_steps = np.mean(steps_per_episode)
    print(f"Average Steps per Successful Episode: {avg_steps:.2f}")

# env.close()