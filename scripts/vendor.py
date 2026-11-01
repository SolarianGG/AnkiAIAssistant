"""Bundle the add-on's dependencies into anki_ai_assistant/vendor for Anki's Python.

The versions come from uv.lock (the project's runtime dependencies, without the dev group).
Wheels are picked for Anki's Python and Windows, not for the Python running this script.

    uv run scripts/vendor.py              # Anki's Python 3.13
    uv run scripts/vendor.py --python 3.14
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = ROOT / "anki_ai_assistant" / "vendor"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", default="3.13", help="Python version of Anki (Help > About; default: 3.13)")
    parser.add_argument("--platform", default="x86_64-pc-windows-msvc", help="uv target platform")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        requirements = Path(tmp) / "requirements.txt"
        subprocess.run(["uv", "export", "--frozen", "--no-dev", "--no-hashes", "--no-emit-project",
                        "--format", "requirements-txt", "--output-file", str(requirements)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        if VENDOR_DIR.exists():
            shutil.rmtree(VENDOR_DIR)
        subprocess.run(["uv", "pip", "install", "--target", str(VENDOR_DIR), "--python-version", args.python,
                        "--python-platform", args.platform, "--only-binary", ":all:", "--no-deps",
                        "-r", str(requirements)],
                       cwd=ROOT, check=True)

    # Console scripts are useless inside Anki
    shutil.rmtree(VENDOR_DIR / "bin", ignore_errors=True)
    for cache in VENDOR_DIR.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    size = sum(f.stat().st_size for f in VENDOR_DIR.rglob("*") if f.is_file())
    print(f"Vendored into {VENDOR_DIR} ({size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
