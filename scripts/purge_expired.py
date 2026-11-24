
import argparse
import time
from pathlib import Path


def purge_expired_files(directories, retention_days, dry_run=True):
    if retention_days < 1:
        raise ValueError("retention_days doit être supérieur à zéro.")

    cutoff = time.time() - retention_days * 86400
    removed = []

    for directory in directories:
        base = Path(directory).resolve()

        if not base.exists():
            continue

        for path in base.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue

            if base not in path.resolve().parents:
                continue

            try:
                if path.stat().st_mtime < cutoff:
                    removed.append(str(path))
                    if not dry_run:
                        path.unlink()
            except OSError:
                continue

    return removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    # Répertoires de dépôts temporaires uniquement.
    directories = ["data/uploads", "data/reports"]

    removed = purge_expired_files(
        directories,
        args.days,
        dry_run=not args.apply,
    )

    print(f"Fichiers concernés : {len(removed)}")
    for name in removed:
        print(name)

    if not args.apply:
        print("Simulation uniquement. Ajouter --apply pour supprimer.")
