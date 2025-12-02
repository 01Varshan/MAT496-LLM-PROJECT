import pandas as pd
import re
import os

# Load dataset
df = pd.read_csv("data/schemes_prepared.csv")

print("Before cleaning:", len(df), "rows")

# Fix bad Unicode — remove weird characters
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.encode("ascii", "ignore").decode("ascii")  # remove corrupted chars
    text = re.sub(r"\s+", " ", text).strip()  # extra spaces cleanup
    return text

for col in ["scheme_name", "details", "benefits", "eligibility", "application", "documents", "tags"]:
    df[col] = df[col].astype(str).apply(clean_text)

# Filter only strong scheme categories
categories_to_keep = [
    "Business & Entrepreneurship",
    "Skills & Employment",
    "Education & Training",
    "Social Welfare & Empowerment"
]

df = df[df["schemeCategory"].isin(categories_to_keep)]

print("After filtering categories:", len(df), "rows")

# Rebuild clean context field
df["context"] = (
    "Scheme Name: " + df["scheme_name"] + "\n" +
    "Details: " + df["details"] + "\n" +
    "Eligibility: " + df["eligibility"] + "\n" +
    "Benefits: " + df["benefits"] + "\n" +
    "Application Process: " + df["application"] + "\n" +
    "Documents Required: " + df["documents"] + "\n" +
    "Tags: " + df["tags"]
)

# Save cleaned version
os.makedirs("data_cleaned", exist_ok=True)
df.to_csv("data_cleaned/schemes_cleaned.csv", index=False)

print("\n🎯 CLEANING DONE!")
print("Saved cleaned dataset at: data_cleaned/schemes_cleaned.csv")
