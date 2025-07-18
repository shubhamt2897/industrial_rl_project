# environments/franka_reach_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from robot_descriptions.loaders.pybullet import load_robot_description

class FrankaReachEnv(gym.Env):
    """
    Custom Gymnasium environment for a Franka Panda robot to reach a target.
    This version includes a time limit and a more robust IK solver.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super(FrankaReachEnv, self).__init__()
        self.render_mode = render_mode

        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.np_random, _ = gym.utils.seeding.np_random()

        # Set a max number of steps per episode for truncation
        self.max_episode_steps = 1000
        self.current_step = 0
        #Model **{3)** addition
        self.last_joint_positions = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the step counter for the new episode
        self.current_step = 0

        p.resetSimulation(self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        p.loadURDF("plane.urdf", [0, 0, -0.1])
        self.robot_id = load_robot_description("panda_description", basePosition=[0, 0, 0], useFixedBase=True)
        
        # Link index for the 'panda_hand' link
        self.end_effector_link_index = 8 

        # Reset all joints to the zero position used till modle3
        #num_joints = p.getNumJoints(self.robot_id)
       # for i in range(num_joints):
          #  p.resetJointState(self.robot_id, i, targetValue=0)


        # --- NEW: RANDOMIZED JOINT POSITIONS as part of adding randomness for model4 ---
        # Instead of resetting to all zeros, reset to a random valid configuration.
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(7): # Only randomize the 7 arm joints
            # Define a sensible range for each joint to start in
            joint_range = p.getJointInfo(self.robot_id, i)[9] - p.getJointInfo(self.robot_id, i)[8]
            random_pos = self.np_random.uniform(low=-joint_range/4, high=joint_range/4)
            p.resetJointState(self.robot_id, i, targetValue=random_pos)
        # Reset the finger joints to a fixed (open) position
        p.resetJointState(self.robot_id, 7, targetValue=0.04)
        p.resetJointState(self.robot_id, 8, targetValue=0.04)
        # --- END OF NEW CODE ---

        # ADD THIS BLOCK to capture initial joint state , model type 2 for updated reward
        joint_states = p.getJointStates(self.robot_id, range(7))
        self.last_joint_positions = [state[0] for state in joint_states]

        # A central and reliable workspace for the target
        #target_pos_x = self.np_random.uniform(0.3, 0.6)
       # target_pos_y = self.np_random.uniform(-0.3, 0.3)
       # target_pos_z = self.np_random.uniform(0.3, 0.6)
       # self.target_pos = np.array([target_pos_x, target_pos_y, target_pos_z])

        # --- NEW, MORE CHALLENGING TARGET ZONE ---
        # This area is further away from the robot's base.
        target_pos_x = self.np_random.uniform(0.5, 0.7)
        target_pos_y = self.np_random.uniform(-0.3, 0.3)
        target_pos_z = self.np_random.uniform(0.5, 0.7)
        self.target_pos = np.array([target_pos_x, target_pos_y, target_pos_z])
        # Create a visual marker for the target
        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 1])
        self.target_id = p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=self.target_pos)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # Apply a small scaling factor to the agent's action
        velocity_command = action * 0.05
        
        current_pose = p.getLinkState(self.robot_id, self.end_effector_link_index)
        current_position = np.array(current_pose[0])
        new_position = current_position + velocity_command
        
        # Define a comfortable "rest pose" for the arm's joints
        rest_poses = [0, -0.785, 0, -2.356, 0, 1.57, 0.785]
        
        # Use the robust IK solver with joint damping for stability
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_link_index,
            new_position,
            restPoses=rest_poses,
            jointDamping=[0.1] * 9
        )
        
        # Command the 7 arm joints
        p.setJointMotorControlArray(
            self.robot_id,
            range(7),
            p.POSITION_CONTROL,
            targetPositions=joint_poses[:7] # Ensure we only take the first 7
        )
        
        p.stepSimulation()
        self.current_step += 1
        
        observation = self._get_obs()
        info = self._get_info()
        distance = np.linalg.norm(info['end_effector_pos'] - self.target_pos)
        info['distance'] = distance # Add distance to info dict
        
        # Define reward and termination conditions
        # 1. Main reward for getting closer to the target
        distance_reward = -distance
        # --- REWARD SHAPING --- Model **{2}**
        # 2. Penalty for being near joint limits to encourage natural poses
       # joint_limit_penalty = 0
       # joint_states = p.getJointStates(self.robot_id, range(7))
       # joint_positions = [state[0] for state in joint_states]
        
        # These are the official joint limits for the Franka Panda
        # lower_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        # upper_limits = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        
        # A simple penalty based on how far joints are from their neutral 'rest_poses'
       # for i in range(len(rest_poses)):
           # joint_limit_penalty += (joint_positions[i] - rest_poses[i])**2
        
        # The final reward is a combination of the two, with a small weight on the penalty
        #reward = distance_reward - 0.01 * joint_limit_penalty 
        # --- END OF REWARD SHAPING ---
         # --- NEW REWARD CALCULATION model 3 ---
        observation = self._get_obs()
        info = self._get_info()
        distance = np.linalg.norm(info['end_effector_pos'] - self.target_pos)
        info['distance'] = distance
        
        # 1. Main reward for getting closer
        distance_reward = -distance

        # 2. Penalty for being near joint limits
        joint_limit_penalty = 0
        joint_states = p.getJointStates(self.robot_id, range(7))
        current_joint_positions = [state[0] for state in joint_states]
        for i in range(len(rest_poses[:7])):
            joint_limit_penalty += (current_joint_positions[i] - rest_poses[i])**2
        
        # 3. NEW: Action Cost Penalty (for minimal movement)
        action_penalty = np.sum(np.square(action))

        # 4. NEW: Movement Variance Penalty (for distributed movement)
        joint_movements = np.abs(np.array(current_joint_positions) - np.array(self.last_joint_positions))
        variance_penalty = np.var(joint_movements)
        
        # Update for next step
        self.last_joint_positions = current_joint_positions

        # Final reward combines all components with different weights
        reward = (
            distance_reward                      # Main goal
            - 0.01 * joint_limit_penalty         # Secondary: prefer neutral pose
            - 0.001 * action_penalty             # Tertiary: prefer small actions
            - 0.1 * variance_penalty             # Tertiary: prefer evenly spread actions
        )
        # --- END OF REWARD CALCULATION ---
        terminated = distance < 0.05
        truncated = self.current_step >= self.max_episode_steps

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        end_effector_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_link_index)[0])
        return np.concatenate([end_effector_pos, self.target_pos]).astype(np.float32)

    def _get_info(self):
        end_effector_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_link_index)[0])
        return {'end_effector_pos': end_effector_pos}

    def close(self):
        p.disconnect(self.client)