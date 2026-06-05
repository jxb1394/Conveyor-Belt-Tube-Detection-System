import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
from stable_baselines3 import SAC
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class ConveyorSortingEnv(gym.Env):
    ###############################################################################################################################################################
    def __init__(self, render=False, use_robot=False):
        super(ConveyorSortingEnv, self).__init__()
        self.TUBE_URDF_PATHS = [
            r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-tube-1.urdf",
            r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-tube-2.urdf",
            r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-centrifugal-1.urdf",
            r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-centrifugal-2.urdf",
            r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\lysis-tube-1.urdf"
        ]
        self.CONVEYOR_2_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/conveyor_belt_with_hopper.urdf"
        self.CONVEYOR_3_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/conveyor_belt_shortened.urdf"
        self.CONVEYOR_4_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/conveyor_belt_project_2.urdf"
        self.AIR_JET_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/air_jet.urdf"
        self.BIN_URDF = "C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/bin.urdf"
        self.BIN_FUNNEL_URDF = "C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/bin_funnel.urdf"
        self.ur5_path = "./urdf/ur5_robotiq_85.urdf"
        
        self.jet_positions = [2.8*i-5.5 for i in range(5)]
        self.action_space = spaces.Box(low=0, high=5.99, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.tubes = []
        self.step_counter = 0
        self.use_robot = use_robot
        self.log_file = open("debug_log.csv", mode="w", newline="")
        self.logger = csv.writer(self.log_file)
        self.logger.writerow(["step", "action", "obs1_t1","obs2_t1", "reward"])
    
    ###############################################################################################################################################################
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        conveyor_id_2 = p.loadURDF(self.CONVEYOR_2_URDF, basePosition=[0, -14, 7], useFixedBase=True)
        p.changeVisualShape(conveyor_id_2, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
        conveyor_id_3 = p.loadURDF(self.CONVEYOR_3_URDF, basePosition=[4, 0.7, 4.2*0.9], useFixedBase=True, globalScaling=0.9)
        p.changeVisualShape(conveyor_id_3, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
        conveyor_id_4 = p.loadURDF(self.CONVEYOR_4_URDF, basePosition=[0, 0.7, 11], useFixedBase=True)       
        p.changeVisualShape(conveyor_id_4, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
        cyl_vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=0.8, length=2, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
        cyl_col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.8, height=2)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cyl_col_id, baseVisualShapeIndex=cyl_vis_id, basePosition=[4.8, 2.8*2-5.5, 1])
        self.robot_unscrewer = p.loadURDF(self.ur5_path, [4.8, 2.8*2-5.5 ,2], useFixedBase=True, globalScaling=6)
        
        
        
        air_jet_ori = p.getQuaternionFromEuler([0, 0, 1.57])
        bin_ori = p.getQuaternionFromEuler([0, 0, 1.57])
        for i, y in enumerate(self.jet_positions):
            air_jet = p.loadURDF(self.AIR_JET_URDF, basePosition=[-3.9, y, 12.6], baseOrientation=air_jet_ori, useFixedBase=True)
            p.changeVisualShape(air_jet, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
            bin = p.loadURDF(self.BIN_FUNNEL_URDF, basePosition=[2.3, y, 8.5], baseOrientation=bin_ori, useFixedBase=True)
            p.changeVisualShape(bin, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
            
        self.tubes = []
        self.step_counter = 0
        self._spawn_tube()
        return self._get_obs(), {}

    ###############################################################################################################################################################
    def _get_obs(self):
        if not self.tubes:
            return np.zeros(2, dtype=np.float32)
        active_tube = max(self.tubes, key=lambda t: p.getBasePositionAndOrientation(t['id'])[0][0])
        pos, _ = p.getBasePositionAndOrientation(active_tube['id'])
        y_norm = pos[1] / 9.0
        type_norm = (active_tube['type'] - 2.0) / 2.0
        return np.array([y_norm, type_norm], dtype=np.float32)

    ###############################################################################################################################################################
    def _spawn_tube(self):
        idx = np.random.randint(0, 5)
        start_ori = p.getQuaternionFromEuler([1.5708, 0, np.random.uniform(0, 6.28)])
        t_id = p.loadURDF(self.TUBE_URDF_PATHS[idx], [-2, -13, 11], start_ori)
        self.tubes.append({'id': t_id, 'type': idx})

    ###############################################################################################################################################################
    def _apply_jet_force(self, jet_index):
        target_jet_y = self.jet_positions[jet_index]
        for tube in self.tubes:
            pos, _ = p.getBasePositionAndOrientation(tube['id'])
            if abs(pos[1] - target_jet_y) < 0.5 and -5.0 < pos[0] < 0 and 12 < pos[2] < 12.5:
                p.resetBaseVelocity(tube['id'], linearVelocity=[2, 0, 1.5])
    
    ###############################################################################################################################################################
    def step(self, action):
        jet_index = int(np.floor(action[0]))
        jet_index = np.clip(jet_index, 0, 5)
        reward = 0
        frame_skip = 60
        for _ in range(frame_skip):
            self.step_counter += 1        
            if action < 5:
                self._apply_jet_force(jet_index)
            p.stepSimulation()
            
            self.robot_should_move = False
            for tube in self.tubes[:]:
                pos, _ = p.getBasePositionAndOrientation(tube['id'])
                if -13.5 < pos[1] < -7 and 3 < pos[2] < 13.9:
                    p.resetBaseVelocity(tube['id'], [0, 1.5, 0.5], [0, 0, 0])
                if -8.0 < pos[1] < 8.0 and 12 < pos[2] < 12.25 and -3 < pos[0] < -1:
                    p.resetBaseVelocity(tube['id'], [0, 2.0, 0], [0, 0, 0])
                if -3 < pos[0] < 3 and 0 < pos[2] < 10:
                    upright_ori = p.getQuaternionFromEuler([0, 0, 0])
                    p.resetBasePositionAndOrientation(tube['id'], [2, pos[1], pos[2]], upright_ori)
                    p.resetBaseVelocity(tube['id'], linearVelocity=[0,0,-9.81], angularVelocity=[0, 0, 0])
                if -1.0 < pos[0] < 3.0 and -8.0 < pos[1] < 8.0 and 4 < pos[2] < 5.4:
                    p.resetBaseVelocity(tube['id'], [0, 1, 0], [0, 0, 0])
                if 1.0 < pos[0] < 7.0 and -2.0 < pos[1] < 2.0 and 4 < pos[2] < 6:
                    self.robot_should_move = True
                    break
            
            if self.step_counter % 500 == 0:
                self._spawn_tube()
        
        # Simple sorting reward logic (truncated for space)
        # ... [Your reward logic here] ...
        
        terminated = self.step_counter > 4100
        return self._get_obs(), reward, terminated, False, {}

# Heatmap function (Updated for SAC)
def plot_policy_heatmap(model):
    y_vals = np.linspace(-1, 1, 100)
    type_vals = np.linspace(-1, 1, 100)
    action_grid = np.zeros((len(type_vals), len(y_vals)))

    for i, t in enumerate(type_vals):
        for j, y in enumerate(y_vals):
            obs = np.array([y, t], dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            action_grid[i, j] = action

    cmap = ListedColormap(["red", "blue", "green", "purple", "orange", "black"])
    plt.figure(figsize=(8, 6))
    plt.imshow(action_grid, extent=[-1, 1, -1, 1], origin='lower', aspect='auto', cmap=cmap)
    plt.colorbar(label="Action (Jet Index)")
    plt.title("SAC Policy Heatmap")
    plt.show()

if __name__ == "__main__":
    env = ConveyorSortingEnv(render=False)
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        buffer_size=100000, 
        learning_starts=1000, 
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_rate=3e-4,
        tensorboard_log="./sac_sorting_logs/"
    )
    
    print("Starting SAC training...")
    model.learn(total_timesteps=50000)
    model.save("sac_sorting_policy_v1")
    plot_policy_heatmap(model)