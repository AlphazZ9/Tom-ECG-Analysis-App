"""
ecg_app.py -- backward-compatible entry point.

The application has been refactored into the `ecg/` package.
This file exists so existing shortcuts / scripts that call
    python ecg_app.py
continue to work unchanged.

To run directly from the package:
    python -m ecg
"""
import sys
import os

# Ensure the parent directory is on sys.path so `import ecg` works
# regardless of how this script is invoked.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from __main__ import main

if __name__ == "__main__":
    main()
