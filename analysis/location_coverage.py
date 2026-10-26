"""Which spots (and pairs of spots) would have caught the most murmurs in CirCor.
Run: python3 analysis/location_coverage.py
"""
import csv
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPOTS = ["AV", "PV", "TV", "MV"]
NAMES = {"AV": "aortic", "PV": "pulmonic", "TV": "tricuspid", "MV": "mitral"}

rows = list(csv.DictReader(open(ROOT / "data/circor/training_data.csv")))
print(f"{len(rows)} children; murmur: {dict(Counter(r['Murmur'] for r in rows))}")

# only kids recorded at all four spots so it's fair
murmur = [r for r in rows if r["Murmur"] == "Present"
          and set(SPOTS) <= set(r["Recording locations:"].split("+"))]
heard = [set(r["Murmur locations"].split("+")) for r in murmur]
print(f"{len(murmur)} murmur children recorded at all four spots\n")

print("Most audible spot:")
for spot, n in Counter(r["Most audible location"] for r in murmur).most_common():
    print(f"  {spot:4} {NAMES.get(spot, ''):10} {n / len(murmur):5.0%}")

print("\nMurmur heard at this spot:")
for s in SPOTS:
    print(f"  {s:4} {NAMES[s]:10} {sum(s in h for h in heard) / len(murmur):5.0%}")

print("\nMurmur heard at one or both spots of a pair:")
pairs = sorted(combinations(SPOTS, 2), key=lambda p: -sum(bool(h & set(p)) for h in heard))
for p in pairs:
    print(f"  {'+'.join(p):6} {sum(bool(h & set(p)) for h in heard) / len(murmur):5.0%}")
