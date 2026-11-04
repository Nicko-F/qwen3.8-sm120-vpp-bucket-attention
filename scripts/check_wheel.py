"""Verify that an installable wheel contains every lazy JIT source asset."""

from pathlib import Path
import sys
import zipfile


def main():
    source = Path(__file__).resolve().parents[1] / "src"
    required = [
        p.relative_to(source).as_posix()
        for p in source.rglob("*")
        if p.suffix in (".cu", ".cuh", ".cpp", ".h")
    ]
    for wheel in sys.argv[1:]:
        with zipfile.ZipFile(wheel) as archive:
            missing = sorted(set(required) - set(archive.namelist()))
        if missing:
            raise RuntimeError(f"{wheel} omits JIT assets: {missing}")
        print(f"{Path(wheel).name}: all {len(required)} JIT source assets included")


if __name__ == "__main__":
    main()
