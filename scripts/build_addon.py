"""Build the installable add-on: dist/anki_ai_assistant-<version>.ankiaddon.

Rebuilds vendor/ for Anki's Python (see vendor.py), then zips the add-on folder. Left out: caches,
the user's settings (meta.json, with API keys) and user_files (downloads, cache, logs).
Install the file with Tools > Add-ons > Install from file, or by double-clicking it.

    uv run scripts/build_addon.py
    uv run scripts/build_addon.py --python 3.14
    uv run scripts/build_addon.py --keep-vendor    # zip the vendor/ that is already there
"""

import argparse
import json
import zipfile
from pathlib import Path

from vendor import DEFAULT_PLATFORM, DEFAULT_PYTHON, ROOT, VENDOR_DIR, build_vendor

ADDON_DIR = ROOT / "anki_ai_assistant"
DIST_DIR = ROOT / "dist"

EXCLUDED_NAMES = {"__pycache__", "meta.json", "user_files", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def addon_files() -> list[Path]:
    files = []
    for path in sorted(ADDON_DIR.rglob("*")):
        relative = path.relative_to(ADDON_DIR)
        if not path.is_file() or any(part in EXCLUDED_NAMES for part in relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return files


def build(python: str, platform: str, keep_vendor: bool) -> Path:
    if not (keep_vendor and VENDOR_DIR.is_dir()):
        build_vendor(python, platform)
    manifest = json.loads((ADDON_DIR / "manifest.json").read_text(encoding="utf-8"))
    DIST_DIR.mkdir(exist_ok=True)
    target = DIST_DIR / f"{manifest['package']}-{manifest['human_version']}.ankiaddon"
    # Anki expects manifest.json at the top of the archive, not inside a folder
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in addon_files():
            archive.write(path, path.relative_to(ADDON_DIR).as_posix())
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="Python version of Anki (Help > About; default: 3.13)")
    parser.add_argument("--platform", default=DEFAULT_PLATFORM, help="uv target platform")
    parser.add_argument("--keep-vendor", action="store_true", help="do not rebuild an existing vendor/")
    args = parser.parse_args()
    target = build(args.python, args.platform, args.keep_vendor)
    print(f"Built {target} ({target.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
