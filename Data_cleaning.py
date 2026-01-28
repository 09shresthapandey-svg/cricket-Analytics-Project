import pandas as pd
import glob
import os

# Load one sample match file
files = glob.glob("data/raw/**/*.csv", recursive=True)
df = pd.read_csv(files[0])

# RAW DATA VIEW
print(df.head())
print(df.isnull().sum())

# Cleaning steps
df_clean = df.dropna()
df_clean.columns = df_clean.columns.str.lower().str.replace(" ", "_")
df_clean = df_clean[df_clean['runs_off_bat'] >= 0]

# Save cleaned data
os.makedirs("data/cleaned", exist_ok=True)
df_clean.to_csv("data/cleaned/cleaned_sample_match.csv", index=False)

print("Cleaned data saved successfully.")
