# train_fetch_rgbd.py

import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import PixelObservationWrapper
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
import os

# --- 1. Custom CNN for RGB-D Data ---
# We create a simple custom network to handle the 4-channel (RGB-D) input.
class RgbdCnn(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # The observation space from the wrapper has a shape of (4, H, W)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute the flattened size once
        with torch.no_grad():
            sample_tensor = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_tensor).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize the RGB channels (first 3), leave depth (channel 4) as is
        observations[:, :3, :, :] = observations[:, :3, :, :] / 255.0
        return self.linear(self.cnn(observations))

# --- 2. Parameters ---
MODEL_NAME = "PPO_FetchReach_RGBD"
LOG_DIR = "logs_fetch"
MODELS_DIR = "trained_models_fetch"
CHECKPOINT_DIR = "checkpoints_fetch"
TIMESTEPS = 2_000_000

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- 3. Create and Wrap the Environment ---
def make_env():
    # Load the standard 'FetchReach-v2' environment
    env = gym.make("FetchReach-v2")
    # Wrap it to get 84x84 RGB-D camera observations.
    # pixels_only=True simplifies the observation to just the image.
    env = PixelObservationWrapper(env, pixels_only=True, camera_name="default", depth=True, render_kwargs={"width": 84, "height": 84})
    return env

# Use the Stable Baselines3 vectorized environment helper
env = make_vec_env(make_env, n_envs=1)


# --- 4. Setup Training ---
checkpoint_callback = CheckpointCallback(
  save_freq=100_000, save_path=CHECKPOINT_DIR, name_prefix=MODEL_NAME
)
policy_kwargs = {
    "features_extractor_class": RgbdCnn,
    "features_extractor_kwargs": dict(features_dim=256),
}

model = PPO(
    "CnnPolicy", env, policy_kwargs=policy_kwargs,
    verbose=1, tensorboard_log=LOG_DIR, device="cuda",
    learning_rate=0.0001, n_steps=2048, batch_size=64,
)

print(f"\n🚀 Starting training for {TIMESTEPS} timesteps on your local machine...")
model.learn(
    total_timesteps=TIMESTEPS, tb_log_name=MODEL_NAME, callback=checkpoint_callback
)

model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}.zip")
model.save(model_path)
print(f"\n----------- ✅ Training Finished -----------")
print(f"Final FetchReach model saved to: {model_path}")

env.close()