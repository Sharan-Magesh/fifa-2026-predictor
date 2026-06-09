path = "src/features/team_features.py"
f = open(path, "r", encoding="utf-8").read()

# Find and replace the entire TEAM_NAME_ALIASES block
import re
pattern = r'TEAM_NAME_ALIASES = \{[^}]+\}'
new_aliases = """TEAM_NAME_ALIASES = {
    "Ivory Coast": "\u00c3\u00b4te d'Ivoire",
    "Cura\u00e7ao":   "Curacao",
    "Czechia":     "Czech Republic",
}"""

# Use the correct Côte character
new_aliases = """TEAM_NAME_ALIASES = {
    "Ivory Coast": "C\u00f4te d'Ivoire",
    "Cura\u00e7ao":   "Curacao",
    "Czechia":     "Czech Republic",
}"""

result = re.sub(pattern, new_aliases, f, flags=re.DOTALL)
if result != f:
    open(path, "w", encoding="utf-8").write(result)
    print("patched")
else:
    print("ERROR: pattern not matched")
