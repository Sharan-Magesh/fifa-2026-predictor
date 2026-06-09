path = "src/simulation/bracket.py"
f = open(path, "r", encoding="utf-8").read()

old = '''GROUPS = {
    "A": ["Mexico",        "South Africa",         "South Korea",    "Czech Republic"],
    "B": ["Canada",        "Bosnia and Herzegovina","Qatar",          "Switzerland"],
    "C": ["Brazil",        "Morocco",               "Haiti",          "Scotland"],
    "D": ["United States", "Paraguay",              "Australia",      "Turkey"],
    "E": ["Germany",       "Belgium",               "Ivory Coast",    "Algeria"],
    "F": ["Sweden",        "Tunisia",               "Netherlands",    "Japan"],
    "G": ["Portugal",      "DR Congo",              "Indonesia",      "New Zealand"],
    "H": ["England",       "Senegal",               "Colombia",       "Norway"],
    "I": ["France",        "Argentina",             "Chile",          "Bosnia and Herzegovina"],
    "J": ["Spain",         "Egypt",                 "Uruguay",        "Iraq"],
    "K": ["Croatia",       "Ecuador",               "Curaçao",        "Ghana"],
    "L": ["Iran",          "Saudi Arabia",          "Jordan",         "Cape Verde"],
}'''

new = '''GROUPS = {
    "A": ["Mexico",       "South Africa",          "South Korea",  "Czechia"],
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
    "L": ["England",      "Croatia",               "Ghana",        "Panama"],
}'''

if old in f:
    f = f.replace(old, new)
    open(path, "w", encoding="utf-8").write(f)
    print("patched")
else:
    print("ERROR: anchor not found")
