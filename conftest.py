"""Root conftest: ensure the project root is on sys.path so that the
``benchmarks`` package can be imported by tests.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
