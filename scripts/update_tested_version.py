"""Update ``tested-versions.json`` with a newly-passed version.

Called by ``.github/workflows/dep-watch.yml`` after a drift test
passes. Updates the version string and the ``_last_updated``
timestamp, leaving everything else (including ``_known_bad``)
untouched.

Usage::

    python scripts/update_tested_version.py <package_name> <new_version>
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSIONS_FILE = REPO_ROOT / "tested-versions.json"


def main(package: str, new_version: str) -> None:
    with VERSIONS_FILE.open() as f:
        data = json.load(f)

    data[package] = new_version
    data["_last_updated"] = date.today().isoformat()

    with VERSIONS_FILE.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")  # POSIX-compliant trailing newline

    print(f"Updated {package} to {new_version}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "usage: update_tested_version.py <package> <version>",
            file=sys.stderr,
        )
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
