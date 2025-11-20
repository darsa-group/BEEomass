import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


SRC_METADATA_PATH = "C:/Users/au738360/PycharmProjects/BEE/metadata_enriched.csv"

df = pd.read_csv(SRC_METADATA_PATH)

biodiscover_S_df = df[(df["DATASET"] == "biodiscover-S") & (df["INSECT_ID"].isin(df["INSECT_ID"].unique()))]
biodiscover_L_df = df[(df["DATASET"] == "biodiscover-L") & (df["INSECT_ID"].isin(df["INSECT_ID"].unique()))]

biodiscover_S_unique = biodiscover_S_df["INSECT_ID"].unique()
biodiscover_L_unique = biodiscover_L_df["INSECT_ID"].unique()
print(biodiscover_L_unique)

biodiscover_S_dist = biodiscover_S_df["ROI_SIZE_MM"].value_counts()
biodiscover_L_dist = biodiscover_L_df["ROI_SIZE_MM"].value_counts()
print(biodiscover_L_dist)

def sample_images(df, column, num_bins=10):
    # Create bins based on the range of the column
    bins = np.linspace(df[column].min(), df[column].max(), num_bins + 1)
    sampled_images = []

    # Iterate through each bin and sample 1 image
    for i in range(len(bins) - 1):
        bin_df = df[(df[column] >= bins[i]) & (df[column] < bins[i + 1])]
        if not bin_df.empty:
            sampled_images.append(bin_df.sample(1))

    # Combine sampled images into a single DataFrame
    return pd.concat(sampled_images)

# Sample images for biodiscover-S and biodiscover-L
biodiscover_S_sampled = sample_images(biodiscover_S_df, "ROI_SIZE_MM")
biodiscover_L_sampled = sample_images(biodiscover_L_df, "ROI_SIZE_MM")

# Display sampled images
print("Sampled images for biodiscover-S:")
print(biodiscover_S_sampled)

print("\nSampled images for biodiscover-L:")
print(biodiscover_L_sampled)

# # Plot histogram for biodiscover-S ROI_SIZE_MM
# plt.hist(biodiscover_S_df["ROI_SIZE_MM"], bins=30, alpha=0.7, color="blue", label="biodiscover-S")
# plt.title("Distribution of ROI_SIZE_MM for biodiscover-S")
# plt.xlabel("ROI_SIZE_MM")
# plt.ylabel("Frequency")
# plt.legend()
# plt.grid(True)
# plt.show()

# Update the SPLIT column to 'train' for the selected samples
biodiscover_S_sampled.loc[:, "SPLIT"] = "train"
biodiscover_L_sampled.loc[:, "SPLIT"] = "train"


print("Updated SPLIT column for biodiscover-S:")
print(biodiscover_S_sampled[["INSECT_ID", "SPLIT"]])

print("\nUpdated SPLIT column for biodiscover-L:")
print(biodiscover_L_sampled[["INSECT_ID", "SPLIT"]])

df.loc[df["INSECT_ID"].isin(biodiscover_S_sampled["INSECT_ID"]), "SPLIT"] = "train"
df.loc[df["INSECT_ID"].isin(biodiscover_L_sampled["INSECT_ID"]), "SPLIT"] = "train"

df.to_csv(SRC_METADATA_PATH, index=False)

