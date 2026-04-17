#!/usr/bin/env python3
"""
One-command reproducibility script for reviewers.
Installs dependencies and runs the full test suite.
"""

import subprocess
import sys

def main():
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

    print("\nRunning regression tests...")
    result = subprocess.run([sys.executable, "-m", "tests.test_pipeline"], check=False)

    if result.returncode == 0:
        print("\n All tests passed. Reproducibility confirmed.")
    else:
        print("\n Some tests failed. Please check the output above.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()