from __future__ import annotations

import argparse
from pathlib import Path


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def human(n: int) -> str:
    units = ['B','KB','MB','GB','TB']
    size = float(n)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.2f} {u}"
        size /= 1024.0
    return f"{size:.2f} TB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/data/home/guoruoxi/code/PseudoEdit3D/outputs/hml3d_pattern_batches')
    args = parser.parse_args()
    root = Path(args.root)
    for path in sorted(root.iterdir() if root.exists() else []):
        if path.is_dir():
            print(path.name, human(dir_size(path)))
        else:
            print(path.name, human(path.stat().st_size))


if __name__ == '__main__':
    main()
