import os
from pathlib import Path

print("CWD:", os.getcwd())
print("YAML absolute path:", Path("yolo12bifpn.yaml").resolve())
