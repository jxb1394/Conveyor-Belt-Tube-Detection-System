import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from ultralytics import YOLO

###############################################################################################################################################################
## Load in YOLO model after training ##
MODEL_PATH = Path(r"C:\Users\jbuam\PycharmProjects\pybullet\runs\detect\train24\weights\best.pt")
TUBE_CLASSES = {
    0: "polypropene tube 1",
    1: "polypropene tube 2",
    2: "polystyrene tube 1",
    3: "polystyrene tube 2",
    4: "lysis tube"
}
if MODEL_PATH.exists():
    model = YOLO(str(MODEL_PATH))
    print(f"LOADED CUSTOM MODEL: {MODEL_PATH}")
else:
    print("WARNING: Custom weights not found. AI will guess using default objects.")
    model = YOLO("yolov8n.pt")

###############################################################################################################################################################
## URDF Paths ##
CONVEYOR_1_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/conveyor_belt_draft.urdf"
CONVEYOR_2_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/conveyor_belt_with_hopper.urdf"
CONVEYOR_3_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/conveyor_belt_shortened.urdf"
CONVEYOR_4_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/conveyor_belt_project_2.urdf"
STAND_URDF = "C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/robot-stand.urdf" 
AIR_JET_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/air_jet.urdf"
BIN_URDF = "C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/bin.urdf"
BIN_FUNNEL_URDF = "C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/bin_funnel.urdf"
TUBE_URDF_PATHS = [
    r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-tube-1.urdf",
    r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-tube-2.urdf",
    r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-centrifugal-1.urdf",
    r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-centrifugal-2.urdf",
    r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\lysis-tube-1.urdf"
    #r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\capsule.urdf",
]
TUBE_BASES = [
    r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\tube_urdfs\polypropene-1-base.urdf",
    r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\tube_urdfs\polypropene-2-base.urdf",
    r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\tube_urdfs\polystyrene-1-base.urdf",
    r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\tube_urdfs\polystyrene-2-base.urdf",
    r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\tube_urdfs\lysis-base.urdf"
]
TUBE_CAPS = [
    r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\tube_urdfs\polypropene-1-cap.urdf",
    r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\tube_urdfs\polypropene-2-cap.urdf",
    r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\tube_urdfs\polystyrene-1-cap.urdf",
    r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\tube_urdfs\polystyrene-2-cap.urdf",
    r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\tube_urdfs\lysis-cap.urdf"
]
ur5_path = "./urdf/ur5_robotiq_85.urdf" 

###############################################################################################################################################################
## Set up Simulation ##
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
air_jet_ori = p.getQuaternionFromEuler([0, 0, 1.57])
bin_ori = p.getQuaternionFromEuler([0, 0, 1.57])

'''
## Camera Adjustment ##
camera_distance = 1.1
camera_yaw = 90
camera_pitch = -45
camera_target = [0.5, 0, 0.6]
p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                                 cameraYaw=camera_yaw,
                                 cameraPitch=camera_pitch,
                                 cameraTargetPosition=camera_target)
'''

###############################################################################################################################################################
## Load in URDFs ##
try:
    #conveyor_id = p.loadURDF(CONVEYOR_1_URDF, basePosition=[0, 0, 7], useFixedBase=True)
    conveyor_id_4 = p.loadURDF(CONVEYOR_4_URDF, basePosition=[0, 0.7, 11], useFixedBase=True)
    p.changeVisualShape(conveyor_id_4, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
    conveyor_id_2 = p.loadURDF(CONVEYOR_2_URDF, basePosition=[0, -14, 7], useFixedBase=True)
    p.changeVisualShape(conveyor_id_2, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
    conveyor_id_3 = p.loadURDF(CONVEYOR_3_URDF, basePosition=[4, 0.7, 4.2*0.9], useFixedBase=True, globalScaling=0.9)
    p.changeVisualShape(conveyor_id_3, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
    robot_unscrewer = p.loadURDF(ur5_path, [4.8, 2.8*2-5.5 ,2], useFixedBase=True, globalScaling=6)
    cyl_col_id = p.createCollisionShape(p.GEOM_CYLINDER, 
                                radius=0.8, 
                                height=2)
    cyl_vis_id = p.createVisualShape(p.GEOM_CYLINDER, 
                             radius=0.8, 
                             length=2,
                             rgbaColor=[0.627451, 0.627451, 0.627451, 1])
    cyl_id = p.createMultiBody(baseMass=0, 
                             baseCollisionShapeIndex=cyl_col_id, 
                             baseVisualShapeIndex=cyl_vis_id,
                             basePosition=[4.8, 2.8*2-5.5, 1]) 
    box_stand_half_extents = [2.1, 1.35, 1]
    box_stand_col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_stand_half_extents)
    box_stand_vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=box_stand_half_extents, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
    box_body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=box_stand_col_id, baseVisualShapeIndex=box_stand_vis_id, basePosition=[0, 9, 1])
    bin_instance = p.loadURDF(BIN_URDF, basePosition=[0, 9, 2], baseOrientation=bin_ori, useFixedBase=True)
    p.changeVisualShape(bin_instance, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
    for i in range(5):
        air_jet_instance = p.loadURDF(AIR_JET_URDF, basePosition=[-3.9, 2.8*i-4.8, 12.6], baseOrientation=air_jet_ori, useFixedBase=True)
        p.changeVisualShape(air_jet_instance, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
        binf_instance = p.loadURDF(BIN_FUNNEL_URDF, basePosition=[2.3, 2.8*i-5.5, 8.5], baseOrientation=bin_ori, useFixedBase=True)
        p.changeVisualShape(binf_instance, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
except:
    print("Conveyor URDF not found, using visual belt only.")

CAM_WIDTH, CAM_HEIGHT = 640, 640
CAM_TARGET = [-2.3, 0, 19]
view_matrix = p.computeViewMatrixFromYawPitchRoll(CAM_TARGET, 2.5, 90, -89.9, 0, 2)
proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0)
tubes = []
step_count = 0

###############################################################################################################################################################
## Spawn Tube Function ##
tube_constraints = {}
def spawn_tube():
    idx = np.random.randint(0, 5)
    start_pos = [-2, -13, 11]
    start_ori = p.getQuaternionFromEuler([1.5708, 0, np.random.uniform(0, 6.28)])
    base_id = p.loadURDF(TUBE_BASES[idx], start_pos, start_ori)
    cap_pos, cap_ori = p.multiplyTransforms(start_pos, start_ori, [0, 0, 0.5], [0, 0, 0, 1])
    cap_id = p.loadURDF(TUBE_CAPS[idx], cap_pos, cap_ori)
    cid = p.createConstraint(
        parentBodyUniqueId=base_id,
        parentLinkIndex=-1,
        childBodyUniqueId=cap_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0.5],
        childFramePosition=[0, 0, 0]
    )
    tube_name = TUBE_CLASSES[idx]
    tubes.append({'base_id': base_id, 'cap_id': cap_id, 'constraint_id': cid, 'type': idx, 'class': tube_name, 'fired': False})
    print(tubes[-1])
    '''
    t_id = p.loadURDF(TUBE_URDF_PATHS[idx], [-2, -13, 11], start_ori)
    p.changeDynamics(t_id, -1, mass=0.2, lateralFriction=2.0, angularDamping=0.9)
    tube_name = TUBE_CLASSES[idx]
    tubes.append({'id': t_id, 'type': idx, 'class': tube_name, 'fired': False})
    #print(tubes[-1])
    '''

###############################################################################################################################################################
## Guidelines (For Debugging Purposes) ##
corners = [
    [0, -13.5, 3],
    [0, -7, 3],
    [0, -7, 13.9],
    [0, -13.5, 13.9]
]
p.addUserDebugLine(corners[0], corners[1], [1, 0, 0], lineWidth=3) # Bottom
p.addUserDebugLine(corners[1], corners[2], [1, 0, 0], lineWidth=3) # Right
p.addUserDebugLine(corners[2], corners[3], [1, 0, 0], lineWidth=3) # Top
p.addUserDebugLine(corners[3], corners[0], [1, 0, 0], lineWidth=3) # Left
p.addUserDebugText("Tube Spawn Zone", [0, (-13.5+-7)/2, 13.9+1], [1,0,0], 1.5)

corners = [
    [1, -2, 5],
    [1, 2, 5],
    [7, 2, 5],
    [7, -2, 5]
]
p.addUserDebugLine(corners[0], corners[1], [0, 0, 1], lineWidth=3) # Bottom
p.addUserDebugLine(corners[1], corners[2], [0, 0, 1], lineWidth=3) # Right
p.addUserDebugLine(corners[2], corners[3], [0, 0, 1], lineWidth=3) # Top
p.addUserDebugLine(corners[3], corners[0], [0, 0, 1], lineWidth=3) # Left

'''
x_min = -3
x_max = -1
y_min = -8
y_max = 8
z_min = 12
z_max = 12.25
corners = [
    [x_min, y_min, z_min],  # 0
    [x_min, y_max, z_min],  # 1
    [x_min, y_max, z_max],  # 2
    [x_min, y_min, z_max],  # 3
    [x_max, y_min, z_min],  # 4
    [x_max, y_max, z_min],  # 5
    [x_max, y_max, z_max],  # 6
    [x_max, y_min, z_max],  # 7
]
# Front face (x_min)
p.addUserDebugLine(corners[0], corners[1], [1,0,0], 3)
p.addUserDebugLine(corners[1], corners[2], [1,0,0], 3)
p.addUserDebugLine(corners[2], corners[3], [1,0,0], 3)
p.addUserDebugLine(corners[3], corners[0], [1,0,0], 3)
# Back face (x_max)
p.addUserDebugLine(corners[4], corners[5], [1,0,0], 3)
p.addUserDebugLine(corners[5], corners[6], [1,0,0], 3)
p.addUserDebugLine(corners[6], corners[7], [1,0,0], 3)
p.addUserDebugLine(corners[7], corners[4], [1,0,0], 3)
# Connect front to back
p.addUserDebugLine(corners[0], corners[4], [1,0,0], 3)
p.addUserDebugLine(corners[1], corners[5], [1,0,0], 3)
p.addUserDebugLine(corners[2], corners[6], [1,0,0], 3)
p.addUserDebugLine(corners[3], corners[7], [1,0,0], 3)
'''

x_min, x_max, y_min, y_max, z_min, z_max = -1, 3, -8, 8, 4, 6
corners = [
    [x_min, y_min, z_min],
    [x_min, y_max, z_min],
    [x_min, y_max, z_max],
    [x_min, y_min, z_max],
    [x_max, y_min, z_min],
    [x_max, y_max, z_min],
    [x_max, y_max, z_max],
    [x_max, y_min, z_max],
]
# Front face (x_min)
p.addUserDebugLine(corners[0], corners[1], [1,0,0], 3)
p.addUserDebugLine(corners[1], corners[2], [1,0,0], 3)
p.addUserDebugLine(corners[2], corners[3], [1,0,0], 3)
p.addUserDebugLine(corners[3], corners[0], [1,0,0], 3)
# Back face (x_max)
p.addUserDebugLine(corners[4], corners[5], [1,0,0], 3)
p.addUserDebugLine(corners[5], corners[6], [1,0,0], 3)
p.addUserDebugLine(corners[6], corners[7], [1,0,0], 3)
p.addUserDebugLine(corners[7], corners[4], [1,0,0], 3)
# Connect front to back
p.addUserDebugLine(corners[0], corners[4], [1,0,0], 3)
p.addUserDebugLine(corners[1], corners[5], [1,0,0], 3)
p.addUserDebugLine(corners[2], corners[6], [1,0,0], 3)
p.addUserDebugLine(corners[3], corners[7], [1,0,0], 3)



###############################################################################################################################################################
## Main Loop ##
try:
    while True:
        p.stepSimulation()
        step_count += 1
              
        
        if step_count % 1000 == 0:
            spawn_tube()

        if step_count % 30 == 0:
            img_data = p.getCameraImage(CAM_WIDTH, CAM_HEIGHT, view_matrix, proj_matrix, shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            rgb = np.reshape(img_data[2], (CAM_HEIGHT, CAM_WIDTH, 4))[:, :, :3]
            results = model.predict(rgb, conf=0.15, verbose=False, imgsz=320)
            
            if len(results) > 0:
                results[0].names = TUBE_CLASSES
                try:
                    annotated_frame = results[0].plot(line_width=2, labels=True)
                    cv2.imshow("Detection Feed", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                except KeyError:
                    cv2.imshow("Detection Feed", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))         
            cv2.waitKey(1)

        # Movement Logic
        robot_should_move = False
        for tube in tubes[:]:
            pos, _ = p.getBasePositionAndOrientation(tube['base_id'])
            cap_pos, _ = p.getBasePositionAndOrientation(tube['cap_id'])
            if -13.5 < pos[1] < -7 and 3 < pos[2] < 13.9:
                p.resetBaseVelocity(tube['base_id'], [0, 1.5, 0.5], [0, 0, 0])
            if -8.0 < pos[1] < 8.0 and 12 < pos[2] < 12.25 and -3 < pos[0] < -1:
                p.resetBaseVelocity(tube['base_id'], [0, 3.0, 0], [0, 0, 0])
            if pos[1] > 9.0:
                p.removeBody(tube['base_id'])
                p.removeBody(tube['cap_id'])
                tubes.remove(tube)
            if -3 < pos[0] < 3 and 0 < pos[2] < 10:
                upright_ori = p.getQuaternionFromEuler([0, 0, 0])
                p.resetBasePositionAndOrientation(tube['base_id'], [2, pos[1], pos[2]], upright_ori)
                p.resetBaseVelocity(tube['base_id'], linearVelocity=[0,0,-9.81], angularVelocity=[0, 0, 0])
            if -1.0 < pos[0] < 3.0 and -8.0 < pos[1] < 8.0 and 4 < pos[2] < 5.4:
                p.resetBaseVelocity(tube['base_id'], [0, 2, 0], [0, 0, 0])
            if 1.0 < pos[0] < 7.0 and -2.0 < pos[1] < 2.0 and 4 < pos[2] < 6:
                robot_should_move = True
                break
        if robot_should_move:
            target_angles = [0, 0, 0, -1.2, -2.5, -3, -2.5] # in radians
            cap_pos, _ = p.getBasePositionAndOrientation(tube['cap_id'])
            for i, angle in enumerate(target_angles):
                p.setJointMotorControl2(
                    bodyIndex=robot_unscrewer,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=angle,
                    force=40000,
                    maxVelocity=6
                )
            gripper_state = p.getLinkState(robot_unscrewer, 7)
            gripper_pos = gripper_state[0]
            gripper_ori = gripper_state[1]
            cap_pos, cap_ori = p.getBasePositionAndOrientation(tube['cap_id'])
            inv_gripper_pos, inv_gripper_ori = p.invertTransform(gripper_pos, gripper_ori)
            local_cap_pos, local_cap_ori = p.multiplyTransforms(inv_gripper_pos, inv_gripper_ori, cap_pos, cap_ori)
            distance = np.linalg.norm(np.array(gripper_pos) - np.array(cap_pos))
            if distance < 0.9:
                p.removeConstraint(tube['constraint_id'])
                p.createConstraint(
                    parentBodyUniqueId=robot_unscrewer,
                    parentLinkIndex=7,
                    childBodyUniqueId=tube['cap_id'],
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[-0.2, 0, 0],
                    parentFrameOrientation=[0, 0, 0, 1],
                    childFramePosition=[0, 0, 0]
                )
        else:
            target_angles = [0, 0, 0, 0, 0, 0, 0]         
            for i, angle in enumerate(target_angles):
                p.setJointMotorControl2(
                    bodyIndex=robot_unscrewer,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=angle,
                    force=40000,
                    maxVelocity=4
                )            
        
        # Air Jet Logic
        jet_x_source = -3.9
        jet_z_level = 12.6
        jet_positions = [2.8*i-5.5+0.7 for i in range(5)]
        if step_count % 200 == 0:
            for y_pos in jet_positions:
                min_pt = [-4.0, y_pos - 0.3, jet_z_level - 0.8]
                max_pt = [-1.0, y_pos + 0.3, jet_z_level + 0.8]
                def draw_box(low, high, color):
                    points = [
                        [low[0], low[1], low[2]], [high[0], low[1], low[2]],
                        [high[0], high[1], low[2]], [low[0], high[1], low[2]],
                        [low[0], low[1], high[2]], [high[0], low[1], high[2]],
                        [high[0], high[1], high[2]], [low[0], high[1], high[2]]
                    ]
                    lines = [
                        [0,1], [1,2], [2,3], [3,0], # Bottom
                        [4,5], [5,6], [6,7], [7,4], # Top
                        [0,4], [1,5], [2,6], [3,7]
                    ]
                    for start, end in lines:
                        p.addUserDebugLine(points[start], points[end], color, lineWidth=1, lifeTime=1000.0)

                draw_box(min_pt, max_pt, [0, 1, 0])
        air_color = [0.9, 0.9, 1.0, 0.6] # Slightly more opaque for visibility
        air_vis_id = p.createVisualShape(
            shapeType=p.GEOM_CAPSULE, 
            radius=0.2,      # Wider for a 'gush' look
            length=1.6,      # Long enough to see the direction
            rgbaColor=air_color
        )
        for tube in tubes[:]:
            pos, _ = p.getBasePositionAndOrientation(tube['base_id'])
            tube_type = tube['type']              
            for jet_index, target_jet_y in enumerate(jet_positions):
                y_hit = abs(pos[1] - target_jet_y) < 0.5
                x_hit = -5.0 < pos[0] < -0
                z_hit = 11.0 < pos[2] < 14.0
                
                if y_hit and x_hit and z_hit:
                    if tube_type == jet_index:                       
                        if 12 < pos[2] < 12.5:
                            tube['fired'] = True
                            p.resetBaseVelocity(tube['base_id'], linearVelocity=[3, 0, 1.5])
                            #p.applyExternalForce(tube['id'], -1,
                            #                       forceObj=[10, -6, 20],
                            #                       posObj=pos,
                            #                       flags=p.WORLD_FRAME
                            #)
                                                 
                        #print(f"JET {jet_index} TRIGGERED: Ejecting {TUBE_CLASSES[tube_type]}")
                        tube['fired'] = True
                        p.addUserDebugLine([jet_x_source, target_jet_y, jet_z_level], 
                                           [pos[0], pos[1], pos[2]], 
                                           lineColorRGB=[1, 0, 0], 
                                           lineWidth=2, 
                                           lifeTime=0.1)
                        break
                
                                   
        time.sleep(1. / 240.)
except KeyboardInterrupt:
    p.disconnect()
    cv2.destroyAllWindows()