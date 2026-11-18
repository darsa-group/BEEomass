import csv
from collections import Counter
from pathlib import Path

FOLDER = Path(
    r"C:\Users\au738360\PycharmProjects\BEEomass\01_biomass_model"
)

INPUT_FILE = FOLDER / "metadata_enriched_with_order.csv"
HELD_OUT_ORDER = "Araneae"
OUTPUT_FILE = FOLDER / f"metadata_leave_out_{HELD_OUT_ORDER}.csv"


with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    columns = list(reader.fieldnames)
    rows = list(reader)

required = {"DATASET", "INSECT_ID", "order", "SPLIT"}
if not required.issubset(columns):
    raise ValueError(f"Missing columns: {sorted(required - set(columns))}")

# Identify specimens by dataset and individual ID.
def specimen_key(row):
    return row["DATASET"].strip(), row["INSECT_ID"].strip()


held_out_specimens = {
    specimen_key(row)
    for row in rows
    if row["order"].strip().casefold() == HELD_OUT_ORDER.casefold()
}

if not held_out_specimens:
    raise ValueError(f"No specimens found for order: {HELD_OUT_ORDER}")

# Move every image of each held-out specimen to test.
held_out_images = 0

for row in rows:
    if specimen_key(row) in held_out_specimens:
        row["SPLIT"] = "test"
        held_out_images += 1

# Verify that no specimen occurs in multiple splits.
specimen_splits = {}

for row in rows:
    specimen_splits.setdefault(specimen_key(row), set()).add(row["SPLIT"])

conflicts = [
    key for key, splits in specimen_splits.items()
    if len(splits) > 1
]

if conflicts:
    raise ValueError(f"Specimens appear in multiple splits: {conflicts[:5]}")

# Exclusive creation prevents overwriting an existing experiment.
with OUTPUT_FILE.open("x", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print(f"Held-out order: {HELD_OUT_ORDER}")
print(f"Specimens assigned to test: {len(held_out_specimens):,}")
print(f"Images assigned to test: {held_out_images:,}")
print(f"Total image rows per split: {dict(Counter(r['SPLIT'] for r in rows))}")
print(f"Saved: {OUTPUT_FILE}")
