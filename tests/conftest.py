import sys
from pathlib import Path

# make sure the project root is importable no matter where pytest is run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
