import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jpeg-restoration-test-mpl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
