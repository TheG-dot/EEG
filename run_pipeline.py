import os
import time
import subprocess
import psutil

def is_process_running(script_name):
    for q in psutil.process_iter(['cmdline']):
        if q.info['cmdline']:
            cmdline = ' '.join(q.info['cmdline'])
            if script_name in cmdline and 'python' in cmdline.lower():
                return True
    return False

def main():
    print("Waiting for 01_build_dataset.py to finish...")
    while is_process_running("01_build_dataset.py"):
        time.sleep(10)
        
    print("01_build_dataset.py has finished. Running Task 3 Analysis & Regression...")
    try:
        subprocess.run(["python", "02_train_regression.py"], check=True)
    except Exception as e:
        print(f"Error running regression: {e}")
        
    print("Starting Optuna Overnight Optimization...")
    try:
        subprocess.run(["python", "03_train_classification.py"], check=True)
    except Exception as e:
        print(f"Error running optimization: {e}")
        
    print("Overnight pipeline completely finished.")

if __name__ == "__main__":
    main()
