"""
Upload processed dataset to Google Drive for Colab training.
Run this after run_preprocess.py completes.

Usage:
    python scripts/upload_to_drive.py --drive-dir /path/to/mounted/drive
"""
import argparse
import shutil
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy processed dataset to Google Drive")
    parser.add_argument(
        "--drive-dir",
        type=Path,
        required=True,
        help="Path to mounted Google Drive folder e.g. /Volumes/GoogleDrive/My Drive/PhishGuard",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.parquet"),
        help="Processed manifest parquet",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/processed/splits"),
        help="Directory with train/validation/test.parquet",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=Path("data/screenshots"),
    )
    parser.add_argument(
        "--pages-dir",
        type=Path,
        default=Path("data/pages"),
    )
    args = parser.parse_args()

    drive = args.drive_dir
    drive.mkdir(parents=True, exist_ok=True)

    # Copy manifest
    if args.manifest.exists():
        dest = drive / "data" / "processed"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.manifest, dest / "manifest.parquet")
        print(f"Copied manifest → {dest}/manifest.parquet")
    else:
        print(f"WARNING: manifest not found at {args.manifest}")

    # Copy splits
    if args.splits_dir.exists():
        dest = drive / "data" / "processed" / "splits"
        dest.mkdir(parents=True, exist_ok=True)
        for f in args.splits_dir.glob("*.parquet"):
            shutil.copy2(f, dest / f.name)
            print(f"Copied {f.name} → {dest}")
    else:
        print(f"WARNING: splits dir not found at {args.splits_dir}")

    # Copy screenshots (only ok ones referenced in manifest)
    try:
        import pandas as pd
        df = pd.read_parquet(args.manifest)
        ok_screenshots = set(df["screenshot_path"].dropna().tolist())
        ok_pages = set(df["text_path"].dropna().tolist())
    except Exception as e:
        print(f"Could not read manifest for filtering: {e}")
        ok_screenshots = None
        ok_pages = None

    dest_ss = drive / "data" / "screenshots"
    dest_ss.mkdir(parents=True, exist_ok=True)
    ss_count = 0
    for f in args.screenshots_dir.glob("*.png"):
        if ok_screenshots is None or str(f) in ok_screenshots or str(f.resolve()) in ok_screenshots:
            shutil.copy2(f, dest_ss / f.name)
            ss_count += 1
    print(f"Copied {ss_count} screenshots → {dest_ss}")

    dest_pg = drive / "data" / "pages"
    dest_pg.mkdir(parents=True, exist_ok=True)
    pg_count = 0
    for f in args.pages_dir.glob("*.txt"):
        if ok_pages is None or str(f) in ok_pages or str(f.resolve()) in ok_pages:
            shutil.copy2(f, dest_pg / f.name)
            pg_count += 1
    print(f"Copied {pg_count} page texts → {dest_pg}")

    print("\nUpload complete.")
    print(f"In Colab, set DATA_DIR = '/content/drive/MyDrive/PhishGuard/data'")
    return 0


if __name__ == "__main__":
    sys.exit(main())