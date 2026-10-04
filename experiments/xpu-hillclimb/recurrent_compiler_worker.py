"""Execute one retained compiler command from a clean stdlib-only interpreter."""
import json
from pathlib import Path
import subprocess
import sys

request = json.loads(Path(sys.argv[1]).read_text())
result = subprocess.run(request["command"], cwd=request["cwd"], capture_output=True)
sys.stdout.buffer.write(result.stdout)
sys.stderr.buffer.write(result.stderr)
Path(sys.argv[2]).write_text(json.dumps({"returncode": result.returncode}))
sys.exit(0 if result.returncode == 0 else 1)
