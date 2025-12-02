import os
import re
import unicodedata
import pandas as pd

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ---------- CONFIG ----------
RAW_CSV_PATH = "data/schemes_prepared.csv"      # your original file
CLEAN_CSV_PATH = "data/schemes_cleaned.csv"     # we create this
FAISS_DIR = "vectordb/faiss_index"


# ---------- TEXT CLEANING ----------
def clean_text(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x)

    # normalise unicode (fix weird â€™ etc.)
    x = unicodedata.normalize("NFKD", x)

    # remove control chars
    x = "".join(ch for ch in x if ch.isprintable())

    # collapse whitespace
    x = re.sub(r"\s+", " ", x).strip()
    return x


def main():
    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {RAW_CSV_PATH}")

    df = pd.read_csv(RAW_CSV_PATH)
    print("Loaded rows from raw CSV:", len(df))

    # Clean important text columns
    text_cols = [
        "scheme_name",
        "details",
        "benefits",
        "eligibility",
        "application",
        "documents",
        "schemeCategory",
        "tags",
        "level",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(clean_text)
        else:
            df[col] = ""

    # (Optional) drop very empty rows
    df = df[df["details"].str.len() > 20]
    print("Rows after dropping empty details:", len(df))

    # Rebuild context column from cleaned parts
    df["context"] = (
        "Scheme Name: " + df["scheme_name"] + "\n"
        "Details: " + df["details"] + "\n"
        "Eligibility: " + df["eligibility"] + "\n"
        "Benefits: " + df["benefits"] + "\n"
        "Application: " + df["application"] + "\n"
        "Documents Required: " + df["documents"] + "\n"
        "Level: " + df["level"] + "\n"
        "Category: " + df["schemeCategory"] + "\n"
        "Tags: " + df["tags"]
    )

    os.makedirs(os.path.dirname(CLEAN_CSV_PATH), exist_ok=True)
    df.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"✅ Cleaned CSV saved to {CLEAN_CSV_PATH}")

    # ---------- BUILD FAISS INDEX ----------
    os.makedirs(os.path.dirname(FAISS_DIR), exist_ok=True)

    texts = df["context"].tolist()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = splitter.create_documents(texts)
    print("Documents created:", len(docs))

    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectordb = FAISS.from_documents(docs, embedder)

    vectordb.save_local(FAISS_DIR)
    print(f"🎯 FAISS index saved to {FAISS_DIR}")


if __name__ == "__main__":
    main()
