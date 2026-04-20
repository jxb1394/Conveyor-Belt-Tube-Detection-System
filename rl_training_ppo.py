import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
from stable_baselines3 import PPO
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
        self.CONVEYOR_1_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/conveyor_belt_draft.urdf"
        self.CONVEYOR_2_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/conveyor_belt_with_hopper.urdf"
        self.CONVEYOR_3_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/conveyor_belt_shortened.urdf"
        self.CONVEYOR_4_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/conveyor_belt_project_2.urdf"
        self.AIR_JET_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/air_jet.urdf"
        self.BIN_URDF = "C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/bin.urdf"
        self.BIN_FUNNEL_URDF = "C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/bin_funnel.urdf"
        self.ur5_path = "./urdf/ur5_robotiq_85.urdf"
        
        self.jet_positions = [2.8*i-5.5 for i in range(5)]
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.tubes = []
        self.step_counter = 0
        self.use_robot = use_robot
        self.log_file = open("debug_log.csv", mode="w", newline="")
        self.logger = csv.writer(self.log_file)

        self.logger.writerow([
            "step", "action",
            "obs1_t1","obs2_t1",
            "reward"
        ])
    
    ###############################################################################################################################################################
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        #p.loadURDF(self.CONVEYOR_1_URDF, basePosition=[0, 0, 7], useFixedBase=True)
        conveyor_id_2 = p.loadURDF(self.CONVEYOR_2_URDF, basePosition=[0, -14, 7], useFixedBase=True)
        p.changeVisualShape(conveyor_id_2, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
        conveyor_id_3 = p.loadURDF(self.CONVEYOR_3_URDF, basePosition=[4, 0.7, 4.2*0.9], useFixedBase=True, globalScaling=0.9)
        p.changeVisualShape(conveyor_id_3, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
        conveyor_id_4 = p.loadURDF(self.CONVEYOR_4_URDF, basePosition=[0, 0.7, 11], useFixedBase=True)
        p.changeVisualShape(conveyor_id_4, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])

        cyl_vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=0.8, length=2, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
        cyl_col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.8, height=2)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cyl_col_id, baseVisualShapeIndex=cyl_vis_id, basePosition=[4.8, 2.8*2-5.5, 1])
        bin_instance = p.loadURDF(self.BIN_URDF, basePosition=[7.5, 2.8*2-5.5, 0,], useFixedBase=True, globalScaling=0.8)
        
        air_jet_ori = p.getQuaternionFromEuler([0, 0, 1.57])
        bin_ori = p.getQuaternionFromEuler([0, 0, 1.57])
        for i, y in enumerate(self.jet_positions):
            air_jet = p.loadURDF(self.AIR_JET_URDF, basePosition=[-3.9, y, 12.6], baseOrientation=air_jet_ori, useFixedBase=True)
            p.changeVisualShape(air_jet, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
            binf_instance = p.loadURDF(self.BIN_FUNNEL_URDF, basePosition=[2.3, y, 8.5], baseOrientation=bin_ori, useFixedBase=True)
            p.changeVisualShape(binf_instance, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
        self.robot_unscrewer = p.loadURDF(self.ur5_path, [4.8, 2.8*2-5.5 ,2], useFixedBase=True, globalScaling=6)
        
        self.tubes = []
        self.step_counter = 0
        self._spawn_tube()
        return self._get_obs(), {}
    '''
    ###############################################################################################################################################################
    def _get_obs(self):
        relevant_tubes = []
        for tube in self.tubes:
            pos, _ = p.getBasePositionAndOrientation(tube['id'])
            if -8.0 <= pos[1] <= 9.0:
                relevant_tubes.append(tube)
        
        active = sorted(relevant_tubes, key=lambda t: p.getBasePositionAndOrientation(t['id'])[0][1], reverse=True)
        final_obs = []
        for i in range(5):
            if i < len(active):
                pos, _ = p.getBasePositionAndOrientation(active[i]['id'])
                final_obs.extend([pos[1]/9.0, (active[i]['type'] - 2.0) / 2.0])
            else:
                final_obs.extend([0.0, 0.0]) # Changed -2.0 to 0.0 for neural net stability
        #print(f"Obs: {final_obs}")
        return np.array(final_obs, dtype=np.float32)
    '''
    
    
    ###############################################################################################################################################################
    def _spawn_tube(self):
        idx = np.random.randint(0, 5)
        start_ori = p.getQuaternionFromEuler([1.5708, 0, np.random.uniform(0, 6.28)])
        t_id = p.loadURDF(self.TUBE_URDF_PATHS[idx], [-2, -13, 11], start_ori)
        p.changeDynamics(t_id, -1, mass=0.2, lateralFriction=2.0, angularDamping=0.9)
        self.tubes.append({'id': t_id, 'type': idx})
    
    ###############################################################################################################################################################
    def step(self, action):
        reward = 0
        frame_skip = 60
        #print(f"Action: {action}")
        obs = self._get_obs()
        #print(obs)
        active_tube = None

        # Find the tube closest to sorting area (x direction)
        if self.tubes:
            active_tube = max(
                self.tubes,
                key=lambda t: p.getBasePositionAndOrientation(t['id'])[0][0]
            )
        
        for _ in range(frame_skip):
            self.step_counter += 1        
            if action < 5:
                self._apply_jet_force(action)
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
            if self.robot_should_move:
                target_angles = [[0, 0, 0, -1.2, -2.5, -3, -2.5], [0, -3.14, 0, -1.2, -2.5, -3, -2.5]]
                delay = 0
                for j in range(2):
                #target_angles = [0, 0, 0, -1.2, -2.5, -3, -2.5] # in radians
                #cap_pos, _ = p.getBasePositionAndOrientation(tube['cap_id'])
                    for i, angle in enumerate(target_angles[j]):
                        p.setJointMotorControl2(
                            bodyIndex=self.robot_unscrewer,
                            jointIndex=i,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=angle,
                            force=40000,
                            maxVelocity=6
                        )
                    
                    #for _ in range(1000):
                    #   p.stepSimulation()
                        
                    gripper_state = p.getLinkState(self.robot_unscrewer, 7)
                    gripper_pos = gripper_state[0]
                    gripper_ori = gripper_state[1]
                    #time.sleep(2)
                #cap_pos, cap_ori = p.getBasePositionAndOrientation(tube['cap_id'])
                #inv_gripper_pos, inv_gripper_ori = p.invertTransform(gripper_pos, gripper_ori)
                #local_cap_pos, local_cap_ori = p.multiplyTransforms(inv_gripper_pos, inv_gripper_ori, cap_pos, cap_ori)
                #distance = np.linalg.norm(np.array(gripper_pos) - np.array(cap_pos))             
            else:
                target_angles = [0, 0, 0, 0, 0, 0, 0]         
                for i, angle in enumerate(target_angles):
                    p.setJointMotorControl2(
                        bodyIndex=self.robot_unscrewer,
                        jointIndex=i,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=angle,
                        force=40000,
                        maxVelocity=4
                    )
                
            if self.step_counter % 500 == 0:
                self._spawn_tube()
        
        
        
        for tube in self.tubes[:]:
            pos, _ = p.getBasePositionAndOrientation(tube['id'])
            '''Reward for relative position of tube and air jet'''
            y_pos = pos[1]
            target_y = self.jet_positions[tube['type']]
            dist = abs(y_pos - target_y)
            reward += 5 * (1 - dist / 10)
            '''Reward for if the tube and jet misclassification'''
            if action < 5:
                if abs(y_pos - self.jet_positions[action]) < 0.5:
                    if action == tube['type']:
                        reward += 5
                    else:
                        reward -= 0.1
            '''Penalising misfiring'''
            if action < 5:
                if abs(y_pos - self.jet_positions[action]) > 1.5:
                    reward -= 0.5
        '''Outcome'''
        if pos[0] > 1.5:
            if dist < 0.7:
                reward += 20
        else:
            reward -= 2
        if self.step_counter % 50 == 0:
            self.logger.writerow([self.step_counter, action, *obs.tolist(), reward])
        terminated = self.step_counter > 4100
        return self._get_obs(), reward, terminated, False, {}

    ###############################################################################################################################################################
    def _apply_jet_force(self, jet_index):
        target_jet_y = self.jet_positions[jet_index]
        for tube in self.tubes:
            pos, _ = p.getBasePositionAndOrientation(tube['id'])
            if abs(pos[1] - target_jet_y) < 0.5 and -5.0 < pos[0] < 0 and 12 < pos[2] < 12.5:
                p.resetBaseVelocity(tube['id'], linearVelocity=[2, 0, 1.5])
                p.addUserDebugLine([-3.9, target_jet_y, 8.6], [pos[0], pos[1], pos[2]], [1, 0, 0], 2, 0.05)

###############################################################################################################################################################
    def _get_obs(self):
        if not self.tubes:
            return np.zeros(2, dtype=np.float32)

        # Get the most relevant tube (closest to decision area)
        active_tube = max(
            self.tubes,
            key=lambda t: p.getBasePositionAndOrientation(t['id'])[0][0]  # closest in x
        )
        pos, _ = p.getBasePositionAndOrientation(active_tube['id'])
        y_norm = pos[1] / 9.0
        type_norm = (active_tube['type'] - 2.0) / 2.0
        final_obs = [y_norm, type_norm]
        return np.array(final_obs, dtype=np.float32)
    
###############################################################################################################################################################
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
    plt.imshow(
        action_grid,
        extent=[-1, 1, -1, 1],
        origin='lower',
        aspect='auto',
        cmap=cmap,
        vmin=0,
        vmax=5
    )
    plt.colorbar(label="Action (Jet Index)")
    plt.xlabel("Tube Y Position (normalized)")
    plt.ylabel("Tube Type (normalized)")
    plt.title("PPO Policy Heatmap (State → Action)")
    plt.show()


if __name__ == "__main__":
    env = ConveyorSortingEnv(render=False)
    # Increased n_steps for better stability with 5 discrete actions
    model = PPO("MlpPolicy", env, learning_rate=3e-4, gamma=0.99, verbose=1, n_steps=2048, tensorboard_log="./ppo_sorting_logs/")
    model.learn(total_timesteps=200000)
    model.save("ppo_sorting_policy_v2")
    plot_policy_heatmap(model)