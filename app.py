import subprocess

subprocess.run(
        ["curl", "--output", "checkpoint.pkl", "https://storage.googleapis.com/ithaca-resources/models/checkpoint_v1.pkl"])
subprocess.run(
        ["python", "inference_example.py", "--help"])
