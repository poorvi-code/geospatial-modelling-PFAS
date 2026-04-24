import subprocess
import sys
from pathlib import Path

# Redirect to the new modular dashboard
main_path = Path(__file__).resolve().parent / "dashboard" / "main.py"

if __name__ == "__main__":
    # If run directly as a script, we can use subprocess to run the actual main.py
    # But Streamlit runs the file. We should actually just import the main logic or exec the file.
    # The simplest way for Streamlit is to exec the target file in the current context.
    with open(main_path, "r") as f:
        code = f.read()
    
    # Update sys.path to allow imports from the dashboard folder
    sys.path.append(str(main_path.parent))
    
    exec(code, globals())
