import pandas as pd
import os

csv_path = "data/schemes_prepared.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError("CSV not found in data/ folder!")

df = pd.read_csv(csv_path)
print("Columns in file:", df.columns)

# Create context column
df["context"] = (
    "Scheme Name: " + df["scheme_name"].astype(str) + "\n\n" +
    "Details: " + df["details"].astype(str) + "\n\n" +
    "Benefits: " + df["benefits"].astype(str) + "\n\n" +
    "Eligibility: " + df["eligibility"].astype(str) + "\n\n" +
    "Application: " + df["application"].astype(str) + "\n\n" +
    "Documents Required: " + df["documents"].astype(str) + "\n\n" +
    "Category: " + df["schemeCategory"].astype(str) + "\n\n" +
    "Tags: " + df["tags"].astype(str)
)

# Remove unwanted columns
if "Unnamed: 9" in df.columns:
    df.drop(columns=["Unnamed: 9"], inplace=True)

df.to_csv(csv_path, index=False)

print("🎯 Context column added successfully!")
print("\nSample context:\n", df["context"].iloc[0][:300], "...")
