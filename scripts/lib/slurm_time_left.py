#!/usr/bin/env python3
"""Turn Slurm's `%L` time-left into an absolute unix deadline.

Stdlib only, Python 3.9 compatible.

Slurm reports remaining wall clock as `[days-]HH:MM:SS`, and sometimes as
`MM:SS` or `UNLIMITED`. The serving job needs the same instant as an absolute
epoch so `/health` can report how long the session has left rather than a
caller discovering the end as a dropped connection.

Prints nothing and exits nonzero on anything it cannot parse, including
`UNLIMITED`. An unbounded session is a real Slurm answer but not a deadline, and
inventing one would report a session as expiring when it is not.

    $ squeue -j 264787 -h -o '%L' | xargs python3 scripts/lib/slurm_time_left.py
    1756312021.4
"""

from __future__ import annotations

import sys
import time
from typing import Optional


def seconds_from_slurm_time_left(value: str) -> Optional[int]:
    """`[days-]HH:MM:SS`, `HH:MM:SS` or `MM:SS` as seconds. None if unparseable."""
    raw = (value or "").strip()
    if not raw or raw.upper() in {"UNLIMITED", "INVALID", "NOT_SET", "N/A"}:
        return None

    days = 0
    if "-" in raw:
        day_part, _, raw = raw.partition("-")
        if not day_part.isdigit():
            return None
        days = int(day_part)

    parts = raw.split(":")
    if not (2 <= len(parts) <= 3) or not all(part.isdigit() for part in parts):
        return None

    numbers = [int(part) for part in parts]
    while len(numbers) < 3:
        numbers.insert(0, 0)
    hours, minutes, seconds = numbers
    if minutes > 59 or seconds > 59:
        return None

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def main(argv: Optional[list] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return 2
    remaining = seconds_from_slurm_time_left(args[0])
    if remaining is None:
        return 1
    now = float(args[1]) if len(args) > 1 else time.time()
    print(now + remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
