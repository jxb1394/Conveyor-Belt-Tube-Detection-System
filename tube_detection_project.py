import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import subprocess
import sys

def main_menu():
    while True:
        print("\n" + "="*30)
        print(" TUBE DETECTION PROJECT MENU")
        print("="*30)
        print("1. Generate New Dataset - Training")
        print("2. Run Training - YOLOv8 Command")
        print("3. Run Live Simulation - Detection")
        print("4. Train RL Agent - PPO")
        print("5. Train RL Agent - SAC")
        print("6. Exit")
        
        choice = input("\nSelect an option (1-5): ")

        if choice == '1':
            from test import run_data_collection
            run_data_collection()
        elif choice == '2':
            print("\nStarting YOLOv8 Training...")
            env = os.environ.copy()
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            subprocess.run(
                "yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=30 imgsz=320 batch=8 workers=0", 
                env=env, 
                shell=True)
            #os.system("set KMP_DUPLICATE_LIB_OK=TRUE && yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640 batch=16 workers=0")
        elif choice == '3':
            subprocess.run([sys.executable, "test_.py"])
        elif choice == '4':
            print("\nStarting RL Training Loop...")
            subprocess.run([sys.executable, "rl_training_ppo_dupe_2.py"])
        elif choice == '5':
            print("\nStarting RL Training Loop...")
            subprocess.run([sys.executable, "rl_trainin_sac.py"])
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()