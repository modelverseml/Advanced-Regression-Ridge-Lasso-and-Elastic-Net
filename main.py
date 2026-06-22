"""Entry point: run the Ridge / Lasso / Elastic Net walkthrough.

    python main.py            # print the walkthrough (numbers only)
    python main.py --plot     # also save diagnostic plots to Images/
"""

import os
import sys

# Make both the package (repo root) and the example helpers importable.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
for path in (REPO_ROOT, EXAMPLES_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import walkthrough  # noqa: E402  (lives in examples/, added to path above)

if __name__ == "__main__":
    walkthrough.main()
