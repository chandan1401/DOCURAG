import os
import sys
import subprocess
from pathlib import Path

def main():
    # 1. Define Project Root
    root = Path(__file__).parent.resolve()

    print("Starting DocuRAG Production Launcher...")

    # 2. Setup Environment Variables
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["DJANGO_SETTINGS_MODULE"] = "backend.config.settings"

    # Check for .env file
    if not (root / ".env").exists():
        print("\n⚠️  WARNING: .env file not found in root!")
        print("Please ensure you have a .env file with GEMINI_API_KEY.")

    try:
        # 3. Run Django Migrations
        print("\nInitializing Database...")
        migration_cmd = [sys.executable, "backend/manage.py", "migrate"]
        subprocess.run(migration_cmd, env=env, check=True)
        print("✅ Database Ready.")

        # 4. Launch Server
        print("\nLaunching DocuRAG Web Server...")
        print("--------------------------------------------------")
        print("URL: http://127.0.0.1:8000")
        print("Library: http://127.0.0.1:8000/")
        print("Ask:     http://127.0.0.1:8000/ask/")
        print("--------------------------------------------------")
        print("Press Ctrl+C to stop the server.\n")

        server_cmd = [sys.executable, "backend/manage.py", "runserver"]
        subprocess.run(server_cmd, env=env, check=True)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Critical Error during startup: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
