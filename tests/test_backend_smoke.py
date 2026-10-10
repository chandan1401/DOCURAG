import os
import subprocess
import requests
import time
import sys

def test_backend_smoke():
    print("Starting backend smoke test...")

    # Ensure we are in the directory where manage.py exists or set PYTHONPATH
    # The app structure is backend/manage.py
    # We assume the script is run from the project root

    backend_dir = os.path.abspath("backend")

    # 1. Try to start the Django server in the background
    # Using manage.py runserver
    print("Starting Django server...")

    # Set PYTHONPATH to include the root directory so 'backend' and 'docurag' modules are found
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + os.path.abspath(".")
    env["DJANGO_SETTINGS_MODULE"] = "config.settings"

    proc = subprocess.Popen(
        [sys.executable, os.path.join(backend_dir, "manage.py"), "runserver", "8000"],
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )

    try:
        # Wait for the server to start
        # We'll poll the health endpoint until it responds or we timeout
        max_retries = 30
        success = False
        for i in range(max_retries):
            try:
                response = requests.get("http://127.0.0.1:8000/api/health/", timeout=2)
                if response.status_code == 200:
                    print(f"Health check successful on attempt {i+1}!")
                    print(f"Response: {response.json()}")
                    success = True
                    break
            except requests.exceptions.ConnectionError:
                print(f"Attempt {i+1}: Server not ready yet...")
                time.sleep(2)

        if not success:
            print("Error: Django server failed to start or health endpoint not reachable.")
            # Capture stderr for debugging
            try:
                stdout, stderr = proc.communicate(timeout=10)
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
            except subprocess.TimeoutExpired:
                print("Timed out while capturing server output.")
            sys.exit(1)

    finally:
        # Terminate the server
        print("Shutting down server...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    test_backend_smoke()
