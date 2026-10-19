"""Run the full Capstone II pipeline end to end:  python capstone2/run_all.py"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["gis_features.py", "build_panel.py", "archetypes.py", "models.py", "leakage_audit.py", "explain.py", "figures.py"]

if __name__ == "__main__":
    t0 = time.time()
    for s in STEPS:
        print(f"\n######## {s} ########", flush=True)
        subprocess.run([sys.executable, str(HERE / s)], cwd=HERE, check=True)
    print(f"\n######## tests ########", flush=True)
    subprocess.run([sys.executable, "-m", "pytest", str(HERE / "tests"), "-q"], check=True)
    print(f"\nPipeline completed in {time.time() - t0:.0f} s")
