"""
main.py
=======
PFAS Geospatial Intelligence Platform — Pipeline Orchestrator
--------------------------------------------------------------
Runs the full pipeline in order:
  1. Data cleaning & golden dataset build  (implementation/clean.py)
  2. Model training with SOTA algorithms   (implementation/train.py)
  3. Spatial hotspot detection             (implementation/hotspot.py)
  4. CCI computation                       (implementation/cci.py)

After this completes, launch the dashboard with:
  streamlit run implementation/dashboard.py
"""

import os
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent


def run_step(label: str, module: str, pbar: tqdm, optional: bool = False):
    pbar.set_description(f"Running {label}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    
    result = subprocess.run(
        [sys.executable, "-m", module],
        env=env,
        cwd=str(ROOT),
    )
    
    if result.returncode != 0:
        if optional:
            tqdm.write(f"  ⚠  Optional step '{label}' failed — continuing.")
            return False
        tqdm.write(f"\n  ✗  Step '{label}' FAILED with exit code {result.returncode}.")
        tqdm.write(f"     Error: {result.stderr}")
        tqdm.write("     Fix the error above, then re-run main.py.\n")
        sys.exit(result.returncode)
    
    pbar.update(1)
    return True


def main():
    print("""
╔══════════════════════════════════════════════════════╗
║       PFAS Risk Intelligence — Pipeline Runner       ║
╚══════════════════════════════════════════════════════╝
""")

    steps = [
        ("Step 1 — Build Golden Dataset",        "implementation.clean",   False),
        ("Step 2 — Train SOTA Models (Optuna)",  "implementation.train",   False),
        ("Step 3 — Detect Spatial Hotspots",     "implementation.hotspot", True),
        ("Step 4 — Compute CCI Index",           "implementation.cci",     True),
    ]

    with tqdm(total=len(steps), unit="step", dynamic_ncols=True) as pbar:
        for label, module, optional in steps:
            run_step(label, module, pbar, optional)

    print("""
╔══════════════════════════════════════════════════════╗
║                 ✅  ALL STEPS COMPLETE               ║
╚══════════════════════════════════════════════════════╝

Launch the dashboard:
  streamlit run implementation/dashboard.py

Or run individual steps:
  python -m implementation.clean
  python -m implementation.train
  python -m implementation.hotspot
""")

if __name__ == "__main__":
    main()
