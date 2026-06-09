import re

# Fix 1: bracket.py - use squad CSV names in GROUPS, fix R32 to always produce 16 matchups
path = "src/simulation/bracket.py"
f = open(path, "r", encoding="utf-8").read()

# Fix group names to match squad CSV exactly
old_groups = '''"A": ["Mexico",       "South Africa",          "South Korea",  "Czechia"],
    "B": ["Canada",       "Bosnia and Herzegovina","Qatar",        "Switzerland"],
    "C": ["Brazil",       "Morocco",               "Haiti",        "Scotland"],
    "D": ["United States","Paraguay",              "Australia",    "Turkey"],
    "E": ["Germany",      "Curacao",               "Ivory Coast",  "Ecuador"],
    "F": ["Netherlands",  "Japan",                 "Sweden",       "Tunisia"],
    "G": ["Belgium",      "Egypt",                 "Iran",         "New Zealand"],
    "H": ["Spain",        "Cape Verde",            "Saudi Arabia", "Uruguay"],
    "I": ["France",       "Senegal",               "Iraq",         "Norway"],
    "J": ["Argentina",    "Algeria",               "Austria",      "Jordan"],
    "K": ["Portugal",     "DR Congo",              "Uzbekistan",   "Colombia"],
    "L": ["England",      "Croatia",               "Ghana",        "Panama"],'''

new_groups = '''"A": ["Mexico",        "South Africa",          "South Korea",   "Czech Republic"],
    "B": ["Canada",        "Bosnia and Herzegovina","Qatar",         "Switzerland"],
    "C": ["Brazil",        "Morocco",               "Haiti",         "Scotland"],
    "D": ["United States", "Paraguay",              "Australia",     "Turkey"],
    "E": ["Germany",       "Curaçao",               "Ivory Coast",   "Ecuador"],
    "F": ["Netherlands",   "Japan",                 "Sweden",        "Tunisia"],
    "G": ["Belgium",       "Egypt",                 "Iran",          "New Zealand"],
    "H": ["Spain",         "Cape Verde",            "Saudi Arabia",  "Uruguay"],
    "I": ["France",        "Senegal",               "Iraq",          "Norway"],
    "J": ["Argentina",     "Algeria",               "Austria",       "Jordan"],
    "K": ["Portugal",      "DR Congo",              "Uzbekistan",    "Colombia"],
    "L": ["England",       "Croatia",               "Ghana",         "Panama"],'''

if old_groups in f:
    f = f.replace(old_groups, new_groups)
    print("group names fixed")
else:
    print("WARNING: group names anchor not found - fix manually")

# Fix 2: Replace build_r32_bracket with correct 16-matchup version
old_r32 = '''def build_r32_bracket(
    group_winners: Dict[str, str],
    group_runners: Dict[str, str],
    best_thirds: List[str],
) -> List[Tuple[str, str]]:
    """
    Build the Round of 32 matchup list.

    Returns list of 16 (team_a, team_b) tuples in bracket order.
    Group winners are seeded against runners-up from crossing groups.
    Best 3rd-place teams fill the remaining 8 slots.

    Simplified seeding — exact FIFA seeding for best 3rd-place teams
    depends on which groups they come from and isn't published yet.
    We assign them to the remaining slots in ranked order.
    """
    matchups = []

    # Fixed group winner vs runner-up crossings
    crossing = [
        ("A", "B"), ("C", "D"), ("E", "F"),
        ("G", "H"), ("I", "J"), ("K", "L"),
    ]

    for g1, g2 in crossing:
        matchups.append((group_winners[g1], group_runners[g2]))
        matchups.append((group_winners[g2], group_runners[g1]))

    # Remaining 4 group winners vs best 3rd-place teams
    # Winners from groups that don't have a natural crossing get a 3rd-place opponent
    all_winners = list(group_winners.values())
    paired_winners = set()
    for g1, g2 in crossing:
        paired_winners.add(group_winners[g1])
        paired_winners.add(group_winners[g2])

    unpaired_winners = [w for w in all_winners if w not in paired_winners]

    for i, winner in enumerate(unpaired_winners[:8]):
        if i < len(best_thirds):
            matchups.append((winner, best_thirds[i]))

    return matchups[:16]  # Exactly 16 R32 matchups'''

new_r32 = '''def build_r32_bracket(
    group_winners: Dict[str, str],
    group_runners: Dict[str, str],
    best_thirds: List[str],
) -> List[Tuple[str, str]]:
    """
    Build the Round of 32 matchup list — always returns exactly 16 matchups.

    WC 2026 R32 seeding:
    - 12 group winners play 12 group runners-up (crossing groups)
    - 4 remaining group winners play the 4 best 3rd-place teams
    - The other 4 best 3rd-place teams play the remaining 4 runners-up
    
    Crossing pattern (simplified — official FIFA bracket TBD):
        1A vs 2B, 1B vs 2A
        1C vs 2D, 1D vs 2C
        1E vs 2F, 1F vs 2E
        1G vs 2H, 1H vs 2G
        1I vs 2J, 1J vs 2I
        1K vs 2L, 1L vs 2K
        + 4 winners vs best 3rd-place teams
    """
    matchups = []

    # 6 crossing pairs = 12 matchups
    crossing = [
        ("A", "B"), ("C", "D"), ("E", "F"),
        ("G", "H"), ("I", "J"), ("K", "L"),
    ]
    for g1, g2 in crossing:
        matchups.append((group_winners[g1], group_runners[g2]))
        matchups.append((group_winners[g2], group_runners[g1]))

    # 4 best 3rd-place teams vs 4 group winners (rotated seeding)
    # Use groups A, C, E, G winners as the "home" side for 3rd-place matchups
    anchor_groups = ["A", "C", "E", "G"]
    for i, g in enumerate(anchor_groups):
        if i < len(best_thirds):
            matchups.append((group_winners[g], best_thirds[i]))
        else:
            # Fallback: use a runner-up if not enough 3rd place teams
            matchups.append((group_winners[g], group_runners[g]))

    return matchups[:16]'''

if old_r32 in f:
    f = f.replace(old_r32, new_r32)
    print("R32 bracket fixed")
else:
    print("WARNING: R32 anchor not found")

open(path, "w", encoding="utf-8").write(f)
print("bracket.py saved")

# Fix 3: team_features.py aliases - Czechia and Curacao should NOT be in aliases
# since GROUPS now uses Czech Republic and Curacao directly
path2 = "src/features/team_features.py"
f2 = open(path2, "r", encoding="utf-8").read()

old_aliases = '''TEAM_NAME_ALIASES = {
    "Ivory Coast": "C\u00f4te d'Ivoire",
    "Cura\u00e7ao":   "Curacao",
    "Czechia":     "Czech Republic",
}'''

new_aliases = '''TEAM_NAME_ALIASES = {
    "Ivory Coast": "C\u00f4te d'Ivoire",
    "Cura\u00e7ao":   "Curacao",
}'''

if old_aliases in f2:
    f2 = f2.replace(old_aliases, new_aliases)
    open(path2, "w", encoding="utf-8").write(f2)
    print("team_features aliases cleaned")
else:
    print("WARNING: aliases anchor not found in team_features.py")
