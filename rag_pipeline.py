import pandas as pd
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings   # Updated import
from langchain_community.vectorstores import FAISS

# Path to cleaned dataset
csv_path = "data_cleaned/schemes_cleaned.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError("❌ CSV not found! Run clean_and_rebuild.py first!")

df = pd.read_csv(csv_path)
print("Loaded rows:", len(df))

if "context" not in df.columns:
    raise ValueError("❌ 'context' column missing! Run clean_and_rebuild.py first!")

os.makedirs("vectordb", exist_ok=True)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
docs = splitter.create_documents(df["context"].tolist())
print("Documents created:", len(docs))

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectordb = FAISS.from_documents(docs, embedder)

# Save index (NO dangerous flag when saving)
vectordb.save_local("vectordb/faiss_index")

print("🎯 Vector DB created successfully at: vectordb/faiss_index/")
