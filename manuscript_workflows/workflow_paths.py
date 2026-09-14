"""Keep archived code separate from operator-supplied inputs and outputs."""
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('SPATIOS2E_DATA_ROOT', CODE_ROOT)).resolve()
