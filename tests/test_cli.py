import subprocess
import sys
import os
from pathlib import Path

def test_ingest_cli():
    """
    End-to-end test for the ingest_dir.py CLI tool.
    """
    # Use a test file already in the repository
    test_file = Path("data/raw/test_doc.txt").resolve()
    if not test_file.exists():
        print(f"Creating dummy test file at {test_file}")
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("This is a test document for CLI verification.")

    print(f"Testing CLI with file: {test_file}")

    # Run the script using the same python interpreter as the test
    cmd = [
        sys.executable,
        "scripts/ingest_dir.py",
        str(test_file),
        "--collection", "test_cli_collection"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print("CLI Output:\n", result.stdout)
        print("CLI Error:\n", result.stderr)

        # Since logging uses stderr by default, check both
        combined_output = result.stdout + result.stderr
        if "Successfully ingested" in combined_output:
            print("SUCCESS: CLI tool executed and reported success.")
        else:
            print("FAILURE: CLI tool executed but success message not found in output.")
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        print(f"FAILURE: CLI tool exited with error code {e.returncode}")
        print("Stdout:\n", e.stdout)
        print("Stderr:\n", e.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_ingest_cli()
