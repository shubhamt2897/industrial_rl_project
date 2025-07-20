# environments/franka_reach_env_mujoco.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
from robot_descriptions.loaders.mujoco import load_robot_description

class FrankaReachEnvMujoco(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Load the robot model from an MJCF file
        self.model = load_robot_description("panda_mjcf")
        self.data = mujoco.MjData(self.model)
        
        self.renderer = None
        if self.render_mode in ["human", "rgb_array"]:
            self.renderer = mujoco.Renderer(self.model, height=84, width=84)

        self.image_size = 84
        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=(self.image_size, self.image_size, 3), dtype=np.uint8),
            'depth': spaces.Box(low=0, high=1, shape=(self.image_size, self.image_size, 1), dtype=np.float32)
        })
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.max_episode_steps = 750
        self.current_step = 0
        
        # ADDED: To track joint movement for reward shaping
        self.last_joint_positions = None
        
        self.end_effector_id = self.model.body('panda_hand').id
        self.actuator_ids = [self.model.actuator(f'panda_joint{i}_actuator').id for i in range(1, 8)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        mujoco.mj_resetData(self.model, self.data)
        qpos = self.data.qpos.copy()
        for i in range(7):
            qpos[i] = np.random.uniform(-0.5, 0.5)
        self.data.qpos[:7] = qpos[:7]
        mujoco.mj_forward(self.model, self.data)

        # ADDED: Capture initial joint state for variance penalty
        self.last_joint_positions = self.data.qpos[:7].copy()

        self.target_pos = np.random.uniform(low=[0.3, -0.3, 0.3], high=[0.6, 0.3, 0.6])
        target_body_id = self.model.body('target').id
        self.model.body_pos[target_body_id] = self.target_pos
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_obs(self):
        if self.renderer is None:
             self.renderer = mujoco.Renderer(self.model, height=84, width=84)
        self.renderer.update_scene(self.data, camera="fixed")
        rgb = self.renderer.render()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        depth_reshaped = np.reshape(depth, (self.image_size, self.image_size, 1)).astype(np.float32)
        return {'rgb': rgb, 'depth': depth_reshaped}

    def _get_info(self):
        hand_pos = self.data.body('panda_hand').xpos
        return {'end_effector_pos': hand_pos}

    def step(self, action):
        velocity_command = action * 0.05
        current_position = self.data.body('panda_hand').xpos
        new_position = current_position + velocity_command

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.end_effector_id)
        jacobian = jacp[:, :7]
        joint_velocities = np.linalg.pinv(jacobian) @ (new_position - current_position)
        self.data.ctrl[self.actuator_ids] = joint_velocities
        mujoco.mj_step(self.model, self.data, nstep=5)
        self.current_step += 1
        
        observation = self._get_obs()
        info = self._get_info()
        distance = np.linalg.norm(info['end_effector_pos'] - self.target_pos)
        
        # --- KEY CHANGE: FULL REWARD SHAPING LOGIC ---
        # 1. Main reward for getting closer
        distance_reward = -distance

        # 2. Penalty for being near joint limits
        rest_poses = [0, -0.785, 0, -2.356, 0, 1.57, 0.785]
        current_joint_positions = self.data.qpos[:7]
        joint_limit_penalty = np.sum((current_joint_positions - rest_poses)**2)
        
        # 3. Action Cost Penalty
        action_penalty = np.sum(np.square(action))

        # 4. Movement Variance Penalty
        joint_movements = np.abs(current_joint_positions - self.last_joint_positions)
        variance_penalty = np.var(joint_movements)
        
        # Update for next step
        self.last_joint_positions = current_joint_positions.copy()

        # Final reward combines all components
        reward = (
            distance_reward
            - 0.01 * joint_limit_penalty
            - 0.001 * action_penalty
            - 0.1 * variance_penalty
        )
        # --- END KEY CHANGE ---
        
        terminated = distance < 0.05
        truncated = self.current_step >= self.max_episode_steps
        
        return observation, reward, terminated, truncated, info

    def close(self):
        if self.renderer:
            self.renderer.close()