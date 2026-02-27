"""Test configuration.

Ensures the project root is on sys.path so `import ths_zero` works
when tests are executed from various working directories.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
