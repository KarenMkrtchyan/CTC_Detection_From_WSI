import pandas as pd

# Path to your gzipped Parquet file
file_path = "./data/processed/embeddings_with_pred.parquet.gzip"

# Load the Parquet file (gzip is handled automatically by Pandas)
df = pd.read_parquet(file_path)

# Show the first few rows
print("\n📊 First 5 rows of the data:")
print(df.head())

# Optionally, check column names and shapes
print("\n🧠 Columns:", df.columns.tolist())
print("🔢 Shape:", df.shape)
