import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import os

def run_data_collection():
    dataset_name = input("Please enter the name of the dataset folder: ")
    DATASET_PATH = dataset_name

    # Create YOLO directory structure
    subdirs = ["images/train", "images/val", "labels/train", "labels/val"]
    for sub in subdirs:
        os.makedirs(os.path.join(DATASET_PATH, sub), exist_ok=True)

    CONVEYOR_URDF = r"C:/Users/jbuam/OneDrive/Documents/University/MECHANICAL ENGINEERING/Final year/Final Year Project (MEng)/conveyor_belt_draft.urdf"
    TUBE_URDF_PATHS = [
        r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-tube-1.urdf",
        r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-tube-2.urdf",
        r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-centrifugal-1.urdf",
        r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\polypropene-centrifugal-2.urdf",
        r"C:\Users\jbuam\OneDrive\Documents\University\MECHANICAL ENGINEERING\Final year\Final Year Project (MEng)\lysis-tube-1.urdf"
    ]

    # SIMULATION PARAMETERS
    p.connect(p.GUI) 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    CAM_WIDTH, CAM_HEIGHT = 640, 640 
    CAM_TARGET = [-2, 0, 8.0]
    view_matrix = p.computeViewMatrixFromYawPitchRoll(CAM_TARGET, 2.5, 0, -89.9, 0, 2)
    proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0)

    def setup_scene():
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        p.loadURDF(CONVEYOR_URDF, basePosition=[0, 0, 7], useFixedBase=True)
    def get_precise_yolo_annotation(tube_id, class_id, view_mat, proj_mat):
        pos, _ = p.getBasePositionAndOrientation(tube_id)
        view_mat_np = np.array(view_mat).reshape(4, 4, order='F')
        proj_mat_np = np.array(proj_mat).reshape(4, 4, order='F')
        world_point = np.array([pos[0], pos[1], pos[2], 1.0])
        camera_point = view_mat_np @ world_point
        ndc_point = proj_mat_np @ camera_point
        
        if ndc_point[3] != 0:
            ndc_point /= ndc_point[3]
        
        x_center = (ndc_point[0] + 1.0) / 2.0
        y_center = (1.0 - ndc_point[1]) / 2.0 
        min_p, max_p = p.getAABB(tube_id)
        w_norm = np.clip((max_p[0] - min_p[0]) / 2.2, 0.04, 0.3)
        h_norm = np.clip((max_p[1] - min_p[1]) / 2.2, 0.04, 0.3)
        return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

    # Initial setup
    setup_scene()
    img_counter = 0
    print("Starting Data Collection. Resetting engine every 100 images to prevent stalling...")  
    try:
        while img_counter < 1000:
            if img_counter % 100 == 0 and img_counter > 0:
                setup_scene()
            target_idx = np.random.randint(0, 5)
            sp_x = -2.0 + np.random.uniform(-0.25, 0.25)
            sp_y = np.random.uniform(-0.4, 0.4)
            start_ori = p.getQuaternionFromEuler([1.57, 0, np.random.uniform(0, 6.28)])      
            t_id = p.loadURDF(TUBE_URDF_PATHS[target_idx], [sp_x, sp_y, 8.15], start_ori)

            for _ in range(25):
                p.stepSimulation()
            split = "train" if img_counter < 800 else "val"
            img_data = p.getCameraImage(
                CAM_WIDTH, CAM_HEIGHT, view_matrix, proj_matrix, 
                shadow=1, lightDirection=[1, 1, 1], 
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            rgb_array = np.reshape(img_data[2], (CAM_HEIGHT, CAM_WIDTH, 4))[:, :, :3]
            cv2.imwrite(os.path.join(DATASET_PATH, f"images/{split}/tube_{img_counter}.jpg"), 
                        cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))

            annotation = get_precise_yolo_annotation(t_id, target_idx, view_matrix, proj_matrix)
            with open(os.path.join(DATASET_PATH, f"labels/{split}/tube_{img_counter}.txt"), "w") as f:
                f.write(annotation + "\n")
            p.removeBody(t_id)
            img_counter += 1
            if img_counter % 50 == 0:
                print(f"Captured {img_counter}/1000 images...")
    except KeyboardInterrupt:
        print("\nManual stop detected.")
    finally:
        p.disconnect()
        print(f"Dataset generation complete. Files saved in: {os.path.abspath(DATASET_PATH)}")

if __name__ == "__main__":
    run_data_collection()