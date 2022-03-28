import subprocess
import sys

subprocess.run(
        [sys.executable, "pip", "install", "."])
subprocess.run(
        [sys.executable, "python", "inference_example.py", "--help"])
