from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _flatten_record(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten only dict-valued fields one level deep.
    (Keeps lists like xyxy_mm / box as lists, which pandas can store.)
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:  # flatten non-empty dicts
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv
        else:
            out[k] = v
    return out


def load_json_tree_to_df(root_dir: str | Path) -> pd.DataFrame:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    rows: List[Dict[str, Any]] = []
    bad_files: List[Path] = []

    for p in root.rglob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)

            # If your files ever contain JSONL or a list of records, handle that too:
            if isinstance(obj, list):
                for rec in obj:
                    if isinstance(rec, dict):
                        row = _flatten_record(rec)
                        row["json_parent_dir"] = str(p.parent)
                        row["json_path"] = str(p)
                        rows.append(row)
            elif isinstance(obj, dict):
                row = _flatten_record(obj)
                row["json_parent_dir"] = str(p.parent)  # parent directory of the JSON file
                row["json_path"] = str(p)               # optional but super useful
                rows.append(row)
            else:
                bad_files.append(p)

        except Exception:
            bad_files.append(p)

    df = pd.DataFrame(rows)

    # Optional: expand list columns (xyxy_mm, box, etc.) into separate numeric columns
    # uncomment if you want them as separate columns in the CSV:
    # for col, n, names in [
    #     ("xyxy_mm", 4, ["x1_mm", "y1_mm", "x2_mm", "y2_mm"]),
    #     ("box",     4, ["box_x1", "box_y1", "box_x2", "box_y2"]),
    # ]:
    #     if col in df.columns:
    #         expanded = pd.DataFrame(df[col].tolist(), columns=names, index=df.index)
    #         df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

    if bad_files:
        print(f"Warning: {len(bad_files)} files could not be parsed as JSON. Example: {bad_files[0]}")

    return df


def main():
    root_dir = "00_data/01_segmented/crops"
    out_csv = "metadata.csv"

    df = load_json_tree_to_df(root_dir)

    df = df.rename(columns={"dpi": "DPI"})

    # Create IMAGE_FILENAME = <json_parent_dir>/<crop>
    df["IMAGE_FILENAME"] = (
            df["json_parent_dir"].astype(str).str.rstrip("/")
            + "/"
            + df["crop"].astype(str)
    )
    df["DATASET"] = "crops"
    df["DRYMASS_MG"] = pd.NA

    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df):,} rows to {out_csv}")


if __name__ == "__main__":
    main()
