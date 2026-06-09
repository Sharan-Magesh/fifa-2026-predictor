"""
fix_api_paths.py
Run from project root: python fix_api_paths.py
Patches parents[4] -> parents[3] in all 4 route files.
"""
from pathlib import Path

ROUTE_FILES = [
    "src/api/routes/homepage.py",
    "src/api/routes/match.py",
    "src/api/routes/team.py",
    "src/api/routes/player.py",
]

OLD = "parents[4]"
NEW = "parents[3]"

root = Path(__file__).resolve().parent

for rel_path in ROUTE_FILES:
    fpath = root / rel_path
    if not fpath.exists():
        print(f"SKIP (not found): {fpath}")
        continue

    original = fpath.read_text(encoding="utf-8")
    if OLD not in original:
        print(f"SKIP (already patched or different): {fpath.name}")
        continue

    patched = original.replace(OLD, NEW)
    fpath.write_text(patched, encoding="utf-8")
    count = original.count(OLD)
    print(f"PATCHED {fpath.name} — replaced {count} occurrence(s)")

print("\nDone. Uvicorn will hot-reload automatically.")
print("\nVerify PROJECT_ROOT resolves correctly:")
from src.api.routes import homepage  # will error loudly if path is still wrong
check_path = homepage.PROJECT_ROOT
print(f"  PROJECT_ROOT = {check_path}")
print(f"  Exists:       {check_path.exists()}")
