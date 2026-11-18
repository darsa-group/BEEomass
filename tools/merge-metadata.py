import csv
from pathlib import Path

FOLDER = Path(
    r"C:\Users\au738360\PycharmProjects\BEEomass\01_biomass_model"
)

TAXONOMIC_FILE = FOLDER / "taxonomic-metadata.csv"
ENRICHED_FILE = FOLDER / "metadata_enriched.csv"
OUTPUT_FILE = FOLDER / "metadata_enriched_with_order.csv"


# Build one order lookup per specimen, within each dataset.
order_lookup = {}

with TAXONOMIC_FILE.open(
    "r", encoding="utf-8-sig", newline=""
) as file:
    for row in csv.DictReader(file):
        key = (row["dataset"].strip(), row["insect_id"].strip())
        order = row["order_final"].strip()

        if not order:
            continue

        if key in order_lookup and order_lookup[key] != order:
            raise ValueError(f"Conflicting orders for specimen: {key}")

        order_lookup[key] = order


# Preserve all existing rows, columns, and their order.
with ENRICHED_FILE.open(
    "r", encoding="utf-8-sig", newline=""
) as file:
    reader = csv.DictReader(file)
    columns = list(reader.fieldnames)
    rows = list(reader)

if "order" in columns:
    raise ValueError("The enriched metadata already contains an 'order' column.")

missing = []

for row in rows:
    key = (row["DATASET"].strip(), row["INSECT_ID"].strip())
    row["order"] = order_lookup.get(key, "")

    if not row["order"]:
        missing.append(key)

if missing:
    raise ValueError(
        f"No order found for {len(missing)} rows. Examples: {missing[:5]}"
    )


# "x" prevents accidentally overwriting an existing output file.
with OUTPUT_FILE.open(
    "x", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.DictWriter(file, fieldnames=columns + ["order"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Added order to {len(rows):,} rows.")
print(f"Saved: {OUTPUT_FILE}")