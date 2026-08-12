#!/usr/bin/env python3
"""Copy the solution .py files into (or back out of) the project tree.

Three of the ten exercises are graded in files rather than in the notebook:
E2 (center.py + test_center.py), E7 (fit.py + test_fit.py) and E9 (run.py).
Those ship as scaffolds that raise NotImplementedError, so pytest fails until
you fill them in. This script is here so the answers can be looked at and
copied in without ever quietly overwriting your own work.

    python solutions/apply_solutions.py --diff      # show what would change
    python solutions/apply_solutions.py --apply     # copy them in (backs up first)
    python solutions/apply_solutions.py --restore   # undo the last --apply

The solution notebook does not use this. It shows the file-based answers as
listings instead, so running it can never touch your files.
"""
import argparse
import difflib
import os
import shutil
import sys

SOLUTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(SOLUTIONS_DIR, "files")
PROJECT_DIR = os.path.abspath(os.path.join(SOLUTIONS_DIR, ".."))
BACKUP_DIR = os.path.join(SOLUTIONS_DIR, ".backup")

# solution file name -> where it goes, relative to the project folder
FILE_MAP = {
    "center.py": os.path.join("src", "week01", "center.py"),
    "fit.py": os.path.join("src", "week01", "fit.py"),
    "test_center.py": os.path.join("tests", "test_center.py"),
    "test_fit.py": os.path.join("tests", "test_fit.py"),
    "run.py": "run.py",
}


def read_lines(path):
    """The file as a list of lines, or an empty list if it is not there."""
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return handle.readlines()


def show_diff():
    changed = 0
    for name, where in FILE_MAP.items():
        current = os.path.join(PROJECT_DIR, where)
        solution = os.path.join(FILES_DIR, name)
        lines = difflib.unified_diff(
            read_lines(current), read_lines(solution),
            fromfile="a/" + where, tofile="b/" + where,
        )
        printed = False
        for line in lines:
            sys.stdout.write(line)
            printed = True
        if printed:
            changed = changed + 1
    print()
    print(f"{changed} of {len(FILE_MAP)} files would change")
    return 0


def apply_solutions(force):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for name, where in FILE_MAP.items():
        target = os.path.join(PROJECT_DIR, where)
        backup = os.path.join(BACKUP_DIR, name)

        if os.path.exists(target):
            if os.path.exists(backup) and not force:
                print(f"there is already a backup at {backup} -- use --force to replace it")
                return 1
            shutil.copy2(target, backup)

        shutil.copy2(os.path.join(FILES_DIR, name), target)
        print(f"applied  {where}")

    print()
    print(f"backups are in {BACKUP_DIR}  --  undo with --restore")
    return 0


def restore():
    if not os.path.exists(BACKUP_DIR):
        print("nothing to restore")
        return 1
    for name, where in FILE_MAP.items():
        backup = os.path.join(BACKUP_DIR, name)
        if os.path.exists(backup):
            shutil.copy2(backup, os.path.join(PROJECT_DIR, where))
            print(f"restored {where}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Copy the E2/E7/E9 solutions in or out.")
    parser.add_argument("--diff", action="store_true", help="show what would change")
    parser.add_argument("--apply", action="store_true", help="copy the solutions in")
    parser.add_argument("--restore", action="store_true", help="undo the last --apply")
    parser.add_argument("--force", action="store_true", help="replace an existing backup")
    args = parser.parse_args()

    if args.diff:
        return show_diff()
    if args.apply:
        return apply_solutions(args.force)
    if args.restore:
        return restore()

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
