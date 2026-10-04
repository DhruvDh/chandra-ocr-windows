"""Graph inner deadline; authenticated run_channel Job Object is REQUIRED."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    # Reuse the owned-child deadline implementation without modifying it.
    runpy.run_path(str(Path(__file__).with_name("recurrent_deadline.py")), run_name="__main__")
