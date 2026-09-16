import time
import subprocess
import datetime
import sys

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Command failed: {cmd}\nError: {result.stderr}")
    return result.stdout

def sync():
    try:
        # Add all files (respecting .gitignore)
        subprocess.run("git add .", shell=True, check=True)
        
        # Check if there are changes to commit
        status = run_cmd("git status --porcelain")
        if not status.strip():
            print(f"[{datetime.datetime.now()}] No changes to commit.")
            return

        # Commit changes
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_cmd(f'git commit -m "Auto-commit: {timestamp}"')
        
        # Push to remote
        run_cmd("git push origin main")
        
        print(f"[{datetime.datetime.now()}] Successfully pushed to GitHub.")
    except Exception as e:
        print(f"[{datetime.datetime.now()}] Sync failed: {e}")

if __name__ == "__main__":
    print("Starting auto_sync.py background process. Syncing every 300 seconds.")
    while True:
        sync()
        time.sleep(300)
