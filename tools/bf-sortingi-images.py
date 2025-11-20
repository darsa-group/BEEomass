import pandas as pd
import os
import shutil

ENRICHED_METADATA ="../metadata_enriched.csv"
ROOT_IMG_DIR = "../"
NEW_IMG_DIR = "../03_bfsorted"
os.makedirs(NEW_IMG_DIR, exist_ok=True)

df = pd.read_csv(ENRICHED_METADATA)
sorted_df = df.sort_values(by="BF_cbrMG_MM")
print(sorted_df)


for index, row in df.iterrows():
    # Generate the new filename
    bf_prefix = int(row["BF_cbrMG_MM"] * 1000)
    original_filename = row["IMAGE_FILENAME"]
    new_filename = f"{bf_prefix}_{os.path.basename(original_filename)}"

    # Define source and destination paths
    src_path = os.path.join(ROOT_IMG_DIR, original_filename)
    dest_path = os.path.join(NEW_IMG_DIR, new_filename)

    # Copy the file to the new directory with the new name
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)
    else:
        print(f"File not found: {src_path}")


## Selected insects for the illustration in the paper
## BF_cbrMG_MM,INSECT_ID
## 0.086, 86_crop_2025-09-03_001_B2_09-41-00_CROPNUMBER_0_UUID_ChangeThisTEMPORARY
## 0.162, 162_crop_2025-05-28_003_D2_14-10-47_CROPNUMBER_0_UUID_ChangeThisTEMPORARY
## 0.206, 206_crop_2025-05-28_001_A1_11-56-07_CROPNUMBER_0_UUID_ChangeThisTEMPORARY
## 0.254, 254_crop_2025-09-02_003_D1_10-23-13_CROPNUMBER_0_UUID_ChangeThisTEMPORARY
## 0.351, 351_crop_2025-09-02_003_C5_10-11-11_CROPNUMBER_0_UUID_ChangeThisTEMPORARY

