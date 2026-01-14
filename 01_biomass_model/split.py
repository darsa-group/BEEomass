import pandas as pd
import numpy as np

# ---- config ----
csv_path = "metadata.csv"
seed = 42  # change for a different random split; keep for reproducibility
proportions = {"train": 0.70, "val": 0.10, "test": 0.20}

# ---- load ----
df = pd.read_csv(csv_path)

# ---- allocate per DATASET based on unique INSECT_ID ----
rng = np.random.default_rng(seed)

def split_one_dataset(g: pd.DataFrame) -> pd.DataFrame:
    # unique specimens in this dataset
    ids = g["INSECT_ID"].dropna().unique()
    ids = rng.permutation(ids)  # random order

    n = len(ids)
    n_train = int(np.floor(proportions["train"] * n))
    n_val   = int(np.floor(proportions["val"] * n))
    # remainder goes to test so totals always match
    n_test  = n - n_train - n_val

    train_ids = set(ids[:n_train])
    val_ids   = set(ids[n_train:n_train + n_val])
    test_ids  = set(ids[n_train + n_val:])

    id_to_split = {**{i: "train" for i in train_ids},
                   **{i: "val"   for i in val_ids},
                   **{i: "test"  for i in test_ids}}

    out = g.copy()
    out["SPLIT"] = out["INSECT_ID"].map(id_to_split)
    return out

df = df.groupby("DATASET", group_keys=False).apply(split_one_dataset)

# optional: save
df.to_csv("file_with_split.csv", index=False)