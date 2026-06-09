import re

path = "src/models/match_outcome.py"
f = open(path, "r", encoding="utf-8").read()

# Find the exact location to insert the filter
# Look for the line after the friendly filter
idx = f.find('df = df[df["tournament_type"] != "friendly"].copy()')
if idx == -1:
    print("ERROR: friendly filter line not found")
    exit()

# Find the end of that line
end_of_line = f.find('\n', idx) + 1

# Insert tournament filter after it
insert = '    if tournament_types:\n        df = df[df["tournament_type"].isin(tournament_types)].copy()\n'

f = f[:end_of_line] + insert + f[end_of_line:]
open(path, "w", encoding="utf-8").write(f)
print("patched")

# Verify
f2 = open(path, "r", encoding="utf-8").read()
print("filter applied:", 'isin(tournament_types)' in f2)
