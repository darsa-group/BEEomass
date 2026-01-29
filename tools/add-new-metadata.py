import os
import re
import pandas as pd

BASE = r"I:/BEE/WP-02"
EXPS = ["beetles", "drosophilas", "spiders"]

METADATA_CSV = "I:/BEE/metadata.csv"
OUT_CSV = "I:/BEE/metadata_new.csv"

# ---------- load existing metadata ----------
meta = pd.read_csv(METADATA_CSV)

existing = set()
for i in range(len(meta)):
    existing.add(str(meta.loc[i, "IMAGE_FILENAME"]))

# ---------- helpers ----------
def get_well(fname):
    # Example: 2026-01-09_14-46-27.A5.jpg -> A5
    base = os.path.splitext(fname)[0]
    return base.split(".")[-1].strip().upper()

def extract_date(text):
    # Always return YYYY-MM-DD if it exists inside text
    m = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if m:
        return m.group(0)
    return text

def extract_date_run_from_drymass_filename(filename):
    # Finds: YYYY-MM-DD_001 in drymass filenames
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{3})", filename)
    if m:
        return m.group(1), m.group(2)
    return None, None

def load_drymass_map_and_runs(exp_root):
    drymass_root = os.path.join(exp_root, "drymass")
    mass_map = {}      # full INSECT_ID -> DRYMASS_MG
    runs_by_date = {}  # date -> sorted list of runs

    if not os.path.isdir(drymass_root):
        return mass_map, runs_by_date

    for root, _, files in os.walk(drymass_root):
        for f in files:
            low = f.lower()
            path = os.path.join(root, f)

            if not (low.endswith(".xlsx") or low.endswith(".csv")):
                continue

            # collect runs from filename
            date, run = extract_date_run_from_drymass_filename(f)
            if date and run:
                if date not in runs_by_date:
                    runs_by_date[date] = []
                if run not in runs_by_date[date]:
                    runs_by_date[date].append(run)

            # read drymass file
            df = pd.read_excel(path) if low.endswith(".xlsx") else pd.read_csv(path)
            df.columns = [str(c).strip().upper() for c in df.columns]

            if "INSECT_ID" not in df.columns or "DRYMASS_MG" not in df.columns:
                continue

            for i in range(len(df)):
                k = str(df.loc[i, "INSECT_ID"]).strip().upper().replace(" ", "")
                v = df.loc[i, "DRYMASS_MG"]

                if k == "" or k == "NAN":
                    continue

                # keep first occurrence only
                if k not in mass_map:
                    mass_map[k] = v

    # sort runs for each date
    for d in runs_by_date:
        runs_by_date[d].sort()

    return mass_map, runs_by_date


# ---------- main ----------
new_rows = []
skipped_no_mass = 0

for exp in EXPS:
    exp_root = os.path.join(BASE, exp)
    mass_map, runs_by_date = load_drymass_map_and_runs(exp_root)

    print("\n====", exp, "====")
    print("drymass IDs loaded:", len(mass_map))

    # scan folders under experiment (skip drymass)
    for top in os.listdir(exp_root):
        if top.lower() == "drymass":
            continue

        top_path = os.path.join(exp_root, top)
        if not os.path.isdir(top_path):
            continue

        # IMPORTANT: date must be only YYYY-MM-DD
        date = extract_date(top)

        # list timestamp folders for that date
        ts_folders = []
        for x in os.listdir(top_path):
            p = os.path.join(top_path, x)
            if os.path.isdir(p):
                ts_folders.append(x)
        ts_folders.sort()

        # get runs for this date from drymass filenames
        runs = runs_by_date.get(date, [])
        ts_to_run = {}

        if len(runs) == 0:
            print("WARNING:", exp, date, "no runs found in drymass filenames")
        else:
            if len(ts_folders) != len(runs):
                print("WARNING:", exp, date, "timestamps =", len(ts_folders), "but drymass runs =", len(runs))
            n = min(len(ts_folders), len(runs))
            for i in range(n):
                ts_to_run[ts_folders[i]] = runs[i]

        # scan images
        for ts in ts_folders:
            ts_path = os.path.join(top_path, ts)
            if not os.path.isdir(ts_path):
                continue

            run = ts_to_run.get(ts)
            if run is None:
                # cannot assign run -> cannot match drymass reliably
                continue

            for well_folder in os.listdir(ts_path):
                well_path = os.path.join(ts_path, well_folder)
                if not os.path.isdir(well_path):
                    continue

                for fname in os.listdir(well_path):
                    low = fname.lower()

                    if not low.endswith(".jpg"):
                        continue

                    # skip thumbnails
                    if low.endswith(".thumb.jpg") or ".thumb." in low:
                        continue

                    image_name = fname
                    if image_name in existing:
                        continue

                    well = get_well(fname)

                    insect_id = f"{date}_{run}_{well}"
                    key = insect_id.strip().upper().replace(" ", "")

                    drymass = mass_map.get(key)

                    # SKIP rows with missing drymass
                    if drymass is None or pd.isna(drymass):
                        skipped_no_mass += 1
                        continue

                    new_rows.append({
                        "IMAGE_FILENAME": image_name,
                        "INSECT_ID": insect_id,
                        "DRYMASS_MG": drymass,
                        "DATASET": "EntoScan",
                        "DPI": 3200,
                        "SPLIT": "train",
                        "IS_VALID": True,
                        "w_estimate": None,
                        "nl_estimate": None,
                        "NOTES": ""
                    })

                    existing.add(image_name)

# save
if len(new_rows) > 0:
    meta = pd.concat([meta, pd.DataFrame(new_rows)], ignore_index=True)

meta.to_csv(OUT_CSV, index=False)
print("\nTOTAL added =", len(new_rows))
print("Skipped (no drymass) =", skipped_no_mass)
print("Saved to:", OUT_CSV)
