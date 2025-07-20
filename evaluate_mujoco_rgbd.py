# evaluate_fetch_rgbd.py

import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import PixelObservationWrapper, RecordVideo
import numpy as np
from stable_baselines3 import PPO
import os

# --- Parameters ---
MODEL_NAME = "PPO_FetchReach_RGBD"
MODEL_PATH = f"trained_models_fetch/{MODEL_NAME}.zip" # Path to the final saved model
RECORDINGS_DIR = "fetch_results"
EPISODES_TO_RUN = 20

# --- Load and Wrap the Environment ---
# The environment MUST be wrapped in the same way as the training env
env = gym.make("FetchReach-v2", render_mode="rgb_array")
env = PixelObservationWrapper(env, pixels_only=True, camera_name="default", depth=True, render_kwargs={"width": 84, "height": 84})

# Add the video recorder wrapper
os.makedirs(RECORDINGS_DIR, exist_ok=True)
env = RecordVideo(env, video_folder=RECORDINGS_DIR, name_prefix=f"{MODEL_NAME}-eval")

# --- Load the trained model ---
# The custom network is automatically loaded from the model file
model = PPO.load(MODEL_PATH, env=env)

# --- Run the evaluation ---
for ep in range(EPISODES_TO_RUN):
    obs, info = env.reset()
    done = False
    print(f"\n--- Starting Episode {ep+1}/{EPISODES_TO_RUN} ---")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = True
print("\n--- Evaluation Finished ---")
print(f"Recordings saved to: {RECORDINGS_DIR}")

env.close()