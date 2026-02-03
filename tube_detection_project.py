import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import subprocess

def main_menu():
    while True:
        print("\n" + "="*30)
        print(" TUBE DETECTION PROJECT MENU")
        print("="*30)
        print("1. Generate New Dataset - Training")
        print("2. Run Training - YOLOv8 Command")
        print("3. Run Live Simulation - Detection")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ")

        if choice == '1':
            from test import run_data_collection
            run_data_collection()
        elif choice == '2':
            # This runs the terminal command directly from Python
            print("\nStarting YOLOv8 Training...")
            env = os.environ.copy()
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            subprocess.run(
                "yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=320 batch=8 workers=0", 
                env=env, 
                shell=True)
            #os.system("set KMP_DUPLICATE_LIB_OK=TRUE && yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640 batch=16 workers=0")
        elif choice == '3':
            from test_ import run_live_simulation
            run_live_simulation()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()