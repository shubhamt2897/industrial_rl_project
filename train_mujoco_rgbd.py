# train_mujoco_rgbd.py

import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
from gymnasium.envs.registration import register
# Import the CheckpointCallback
from stable_baselines3.common.callbacks import CheckpointCallback

# --- Register the NEW MUJOCO environment ---
register(
     id="FrankaReachMujocoRGBD-v0",
     entry_point="environments.franka_reach_env_mujoco:FrankaReachEnvMujoco",
)

# --- The Custom Feature Extractor for RGB-D remains THE SAME ---
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        rgb_space = observation_space['rgb']
        extractors['rgb'] = nn.Sequential(
            nn.Conv2d(rgb_space.shape[2], 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample_rgb = torch.as_tensor(rgb_space.sample()[None]).permute(0, 3, 1, 2).float()
            n_flatten_rgb = extractors['rgb'](sample_rgb).shape[1]
        depth_space = observation_space['depth']
        extractors['depth'] = nn.Sequential(
            nn.Conv2d(depth_space.shape[2], 16, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample_depth = torch.as_tensor(depth_space.sample()[None]).permute(0, 3, 1, 2).float()
            n_flatten_depth = extractors['depth'](sample_depth).shape[1]
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = n_flatten_rgb + n_flatten_depth

    def forward(self, observations: dict) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            obs_tensor = observations[key].permute(0, 3, 1, 2)
            encoded_tensor_list.append(extractor(obs_tensor))
        return torch.cat(encoded_tensor_list, dim=1)

# --- Parameters ---
MODEL_NAME = "PPO_Franka_Mujoco_RGBD"
LOG_DIR = "logs_mujoco"
MODELS_DIR = "trained_models_mujoco"
CHECKPOINT_DIR = "checkpoints_mujoco"
TIMESTEPS = 2_000_000 

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Instantiate the MuJoCo Environment ---
env = gym.make("FrankaReachMujocoRGBD-v0", render_mode=None)

# --- Setup the Checkpoint Callback ---
checkpoint_callback = CheckpointCallback(
  save_freq=100_000,
  save_path=CHECKPOINT_DIR,
  name_prefix=MODEL_NAME
)

policy_kwargs = {"features_extractor_class": CustomCombinedExtractor}

model = PPO(
    "MultiInputPolicy", env, policy_kwargs=policy_kwargs,
    verbose=1, tensorboard_log=LOG_DIR, device="cuda",
    learning_rate=0.0001, n_steps=1024, batch_size=64, ent_coef=0.01,
)

# --- Train the model with the callback ---
model.learn(
    total_timesteps=TIMESTEPS, 
    tb_log_name=MODEL_NAME,
    callback=checkpoint_callback
)

model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}.zip")
model.save(model_path)
print(f"\nMuJoCo model saved to: {model_path}")

env.close()