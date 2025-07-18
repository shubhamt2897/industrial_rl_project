# Model Evaluation Report

This document tracks the performance of different versions of the PPO agent trained for the Franka Reach task.

---
## Model Details & Results

### Model 1: Baseline (Simple Reward)

* **Reward Function:** `reward = distance_reward - joint_limit_penalty`
* **Goal:** To teach the agent the basic reaching task while encouraging it to avoid its physical joint limits.
* **Result Interpretation:** This model achieved a moderate success rate (~45%). However, the observed behavior was suboptimal. The agent learned the "laziest" strategy, which involved pointing its wrist at the target and stretching its arm in a straight, unnatural line.

| Metric | Model 1 Results |
| :--- | :--- |
| **Success Rate** | ~45% |
| **Avg. Steps to Success** | ~223 |
| **Observed Behavior** | Awkward "stretching" motion |
| **Training Timesteps** | 1,000,000 |

---

### Model 2: Complex Reward Shaping

* **Reward Function:** `reward = distance_reward 
* **Goal:** To explicitly teach the agent to reach the goal with least distance
* **Result Interpretation:** The success rate is Ok but the robot posture was awkward with only limited joints being exploited for reaching the object

| Metric | Model 2 Results |
| :--- | :--- |
| **Success Rate** | 20.00% |
| **Avg. Steps to Success** | 93.75 |
| **Observed Behavior** | Less fluid, and less consistent |
| **Training Timesteps** | 1,000,000 |

---

### Model 3: [Your Next Experiment Name]

* **Reward Function:** `reward = distance_reward - limit_penalty - action_penalty - variance_penalty`
* **Goal:** To explicitly teach the agent to use **minimal and equally distributed** joint movements, forcing it to learn more natural, fluid motions instead of just stretching.
* **Result Interpretation:** The success rate dropped because the task became significantly harder. However, the successful episodes were completed in less than half the time, proving the agent learned a more efficient and intelligent policy. The model is on the right track but needs more training to master the harder task.

| Metric | Model 3 Results |
| :--- | :--- |
| **Success Rate** | **30.00%** |
| **Avg. Steps to Success** | **87.00** |
| **Observed Behavior** | Highly efficient but still not using all joints. |
| **Training Timesteps** | *2000000* |


### Model 4: Exploration-Focused Training (Converged)

* **What Changed:** Same environment and algorithm as Model 3, but with significantly more training time to allow the agent to master the complex, randomized task.
* **Goal:** To achieve a high success rate on the randomized task, proving the agent has learned a robust and general policy.
* **Result Interpretation:** A major success. The success rate has now surpassed our original baseline, demonstrating the power of the exploration strategies. The agent is now capable of solving the task from a wide variety of starting positions. The higher step count compared to Model 2 is expected and acceptable, as the randomized start positions create a harder problem on average.

| Metric                  | Model 4 Results                                |
| :---------------------- | :--------------------------------------------- |
| **Success Rate** | **55.00%** |
| **Avg. Steps to Success** | **217.55** |
| **Observed Behavior** | *The Rest position and  robots reaching was improoved and the overall robot movement became more natural and evenly dstributed.* |
| **Training Timesteps** | 2,000,000                                     |


### Model 5: Exploration-Focused Training

* **What Changed:** The environment was modified to reset the robot to a **random starting joint configuration** for each episode. The PPO algorithm was also tuned with an **entropy bonus** (`ent_coef=0.01`).
* **Goal:** To force the agent to explore and learn a more general policy that doesn't rely on a single, repetitive motion.
* **Result Interpretation:** A definitive success. After sufficient training time, the combination of randomized starting positions and encouraged exploration proved to be the key. The agent achieved a very high success rate while also being highly efficient. This demonstrates that the agent now has a robust and general understanding of the task.

| Metric                  | Model 4.1 (5) Results                                    |
| :---------------------- | :------------------------------------------------- |
| **Success Rate** | **90.00%** |
| **Avg. Steps to Success** | **137.61** |
| **Observed Behavior** | Confident and successful from most start poses. |
| **Training Timesteps** | 5,000,000                                         |

---
## Overall Comparison Summary

| Metric | Model 1 (Baseline) | Model 2 (Better Reward) | Model 3 (Complex Reward) |
| :--- | :--- | :--- | :--- |
| **Success Rate** | ~45% | 20.00% | 30%|
| **Avg. Steps to Success** | ~223 | 93.75 | 87.00 |

---