import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from ultralytics import YOLO

def run_live_simulation():
    MODEL_PATH = Path(r"C:\Users\jbuam\PycharmProjects\pybullet\runs\detect\train21\weights\best.pt")

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

    # Paths
    CONVEYOR_1_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/conveyor_belt_draft.urdf"
    CONVEYOR_2_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/conveyor_belt_with_hopper.urdf"
    STAND_URDF = "C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/robot-stand.urdf" 
    TUBE_URDF_PATHS = [
        r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-tube-1.urdf",
        r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-tube-2.urdf",
        r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-centrifugal-1.urdf",
        r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-centrifugal-2.urdf",
        r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\lysis-tube-1.urdf"
    ]
    ROBOT_SCALE = 4
    ur5_path = "./urdf/ur5_robotiq_85.urdf" 
    EE_IDX = 11
    ped1_pos = [-5.5, 7.8, 0]
    ped2_pos = [1.5, 7.8, 0]
 
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    # Load Conveyors and robot arms
    try:
        conveyor_id = p.loadURDF(CONVEYOR_1_URDF, basePosition=[0, 0, 7], useFixedBase=True)
        conveyor_id_2 = p.loadURDF(CONVEYOR_2_URDF, basePosition=[0, -14, 7], useFixedBase=True)
        p.changeVisualShape(conveyor_id_2, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])
        robot_holder = p.loadURDF(ur5_path, [ped1_pos[0], ped1_pos[1], 8.2], useFixedBase=True, globalScaling=ROBOT_SCALE)
        p.loadURDF(STAND_URDF, useFixedBase=True, basePosition=ped1_pos)
        p.loadURDF(STAND_URDF, useFixedBase=True, basePosition=ped2_pos)
        robot_unscrewer = p.loadURDF(ur5_path, [ped2_pos[0], ped2_pos[1], 8.2], useFixedBase=True, globalScaling=ROBOT_SCALE)
    except:
        print("Conveyor URDF not found, using visual belt only.")

    CAM_WIDTH, CAM_HEIGHT = 640, 640
    CAM_TARGET = [-2, 0, 8.0]
    view_matrix = p.computeViewMatrixFromYawPitchRoll(CAM_TARGET, 2.5, 0, -89.9, 0, 2)
    proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0)
    tubes = []
    step_count = 0
    def spawn_tube():
        idx = np.random.randint(0, 5)
        start_ori = p.getQuaternionFromEuler([1.5708, 0, np.random.uniform(0, 6.28)])
        t_id = p.loadURDF(TUBE_URDF_PATHS[idx], [-2, -13, 11], start_ori)
        p.changeDynamics(t_id, -1, mass=0.2, lateralFriction=2.0, angularDamping=0.9)
        tubes.append({'id': t_id})

    try:
        while True:
            p.stepSimulation()
            step_count += 1
            
            if step_count % 500 == 0:
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
            for tube in tubes[:]:
                pos, _ = p.getBasePositionAndOrientation(tube['id'])
                if -13.5 < pos[1] < -7.5 and 3 < pos[2] < 13.4:
                    p.resetBaseVelocity(tube['id'], [0, 1.5, 0.5], [0, 0, 0])
                if -8.0 < pos[1] < 8.0 and 8 < pos[2] < 8.25:
                    p.resetBaseVelocity(tube['id'], [0, 3.0, 0], [0, 0, 0])
                if pos[1] > 9.0:
                    p.removeBody(tube['id'])
                    tubes.remove(tube)
            time.sleep(1. / 240.)
    except KeyboardInterrupt:
        p.disconnect()
        cv2.destroyAllWindows()