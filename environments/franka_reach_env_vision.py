# environments/franka_reach_env_vision.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from robot_descriptions.loaders.pybullet import load_robot_description

# RENAMED the class to be specific
class FrankaReachEnvVision(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super(FrankaReachEnvVision, self).__init__()
        self.render_mode = render_mode

        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)

        # The observation is an 84x84 RGB image.
        self.image_size = 84
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.image_size, self.image_size, 3), dtype=np.uint8
        )
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.np_random, _ = gym.utils.seeding.np_random()
        self.max_episode_steps = 750
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        p.resetSimulation(self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf", [0, 0, -0.1])
        self.robot_id = load_robot_description("panda_description", basePosition=[0, 0, 0], useFixedBase=True)
        self.end_effector_link_index = 8
        
        # Randomized joint positions
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(7):
            joint_range = p.getJointInfo(self.robot_id, i)[9] - p.getJointInfo(self.robot_id, i)[8]
            random_pos = self.np_random.uniform(low=-joint_range/4, high=joint_range/4)
            p.resetJointState(self.robot_id, i, targetValue=random_pos)
        p.resetJointState(self.robot_id, 7, targetValue=0.04)
        p.resetJointState(self.robot_id, 8, targetValue=0.04)
        
        joint_states = p.getJointStates(self.robot_id, range(7))
        self.last_joint_positions = [state[0] for state in joint_states]
        
        # Target position
        target_pos_x = self.np_random.uniform(0.5, 0.7)
        target_pos_y = self.np_random.uniform(-0.3, 0.3)
        target_pos_z = self.np_random.uniform(0.5, 0.7)
        self.target_pos = np.array([target_pos_x, target_pos_y, target_pos_z])
        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 1])
        self.target_id = p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=self.target_pos)
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_obs(self):
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0, 0, 1.2],
            cameraTargetPosition=[0.5, 0, 0.2],
            cameraUpVector=[0, 1, 0]
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60.0, aspect=1.0, nearVal=0.1, farVal=3.1
        )
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.image_size, height=self.image_size,
            viewMatrix=view_matrix, projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.array(rgb_img, dtype=np.uint8)
        rgb_frame = np.reshape(rgb_array, (self.image_size, self.image_size, 4))
        return rgb_frame[:, :, :3]

    def _get_info(self):
        end_effector_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_link_index)[0])
        return {'end_effector_pos': end_effector_pos}

    def step(self, action):
        velocity_command = action * 0.05
        current_pose = p.getLinkState(self.robot_id, self.end_effector_link_index)
        current_position = np.array(current_pose[0])
        new_position = current_position + velocity_command
        rest_poses = [0, -0.785, 0, -2.356, 0, 1.57, 0.785, 0.0, 0.0]
        joint_damping = [0.1] * 9
        joint_poses = p.calculateInverseKinematics(
            self.robot_id, self.end_effector_link_index, new_position,
            restPoses=rest_poses, jointDamping=joint_damping
        )
        p.setJointMotorControlArray(
            self.robot_id, range(7), p.POSITION_CONTROL,
            targetPositions=joint_poses[:7]
        )
        p.stepSimulation()
        self.current_step += 1
        
        observation = self._get_obs()
        info = self._get_info()
        distance = np.linalg.norm(info['end_effector_pos'] - self.target_pos)
        info['distance'] = distance
        
        # Reward function
        distance_reward = -distance
        joint_limit_penalty = 0
        joint_states = p.getJointStates(self.robot_id, range(7))
        current_joint_positions = [state[0] for state in joint_states]
        for i in range(len(rest_poses[:7])):
            joint_limit_penalty += (current_joint_positions[i] - rest_poses[i])**2
        action_penalty = np.sum(np.square(action))
        joint_movements = np.abs(np.array(current_joint_positions) - np.array(self.last_joint_positions))
        variance_penalty = np.var(joint_movements)
        self.last_joint_positions = current_joint_positions
        reward = (
            distance_reward
            - 0.01 * joint_limit_penalty
            - 0.001 * action_penalty
            - 0.1 * variance_penalty
        )
        
        terminated = distance < 0.05
        truncated = self.current_step >= self.max_episode_steps
        return observation, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self.client)