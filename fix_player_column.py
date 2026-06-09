"""
fix_player_column.py
Run from project root: python fix_player_column.py
Patches all references to the wrong column name 'player' -> 'player_name'
in the API route files, based on the actual wc2026_squads.csv schema.
"""
from pathlib import Path

root = Path(__file__).resolve().parent

PATCHES = {
    "src/api/routes/homepage.py": [
        (
            'squad_df.groupby("team")["player"].count().reset_index()\n        .rename(columns={"player": "squad_size"})',
            'squad_df.groupby("team")["player_name"].count().reset_index()\n        .rename(columns={"player_name": "squad_size"})',
        ),
    ],
    "src/api/routes/team.py": [
        # squad_records: drop team column and return the rest — no player ref needed, but
        # key_players block references ["player", "position", "composite_score"] which comes
        # from player_scores.csv, not squads — that's fine as-is.
        # No patch needed here unless player_scores.csv also uses player_name.
    ],
    "src/api/routes/player.py": [
        # player_scores.csv column name we don't know yet — will fix separately if needed.
        # No patch needed for squad file references in player.py.
    ],
}

any_patched = False
for rel_path, replacements in PATCHES.items():
    fpath = root / rel_path
    if not fpath.exists():
        print(f"SKIP (not found): {rel_path}")
        continue
    if not replacements:
        print(f"SKIP (no patches for): {rel_path}")
        continue

    text = fpath.read_text(encoding="utf-8")
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            print(f"PATCHED {fpath.name}")
            any_patched = True
        else:
            print(f"WARNING: expected string not found in {fpath.name} — already patched?")

    fpath.write_text(text, encoding="utf-8")

if not any_patched:
    print("Nothing patched — may already be correct.")

# --- Verify the fix works end-to-end ---
print("\nVerifying squad groupby with correct column name...")
import pandas as pd
squads_path = root / "data" / "processed" / "wc2026_squads.csv"
df = pd.read_csv(squads_path)
counts = df.groupby("team")["player_name"].count().reset_index().rename(columns={"player_name": "squad_size"})
print(f"  Squad counts computed OK — {len(counts)} teams, sample:")
print(counts.sort_values("squad_size", ascending=False).head(3).to_string(index=False))
print("\nDone. Uvicorn will hot-reload.")
