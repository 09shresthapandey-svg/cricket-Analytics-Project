import pandas as pd
import glob
import os
import requests
import zipfile

url = "https://cricsheet.org/downloads/t20s_csv2.zip"
zip_path = "t20s_csv2.zip"

# Create raw data folder
os.makedirs("data/raw", exist_ok=True)

print("Downloading T20 matches dataset from Cricsheet...")
r = requests.get(url)
with open(zip_path, 'wb') as f:
    f.write(r.content)


with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("data/raw/")

print("Data extracted to 'data/raw/' folder.")


all_files = glob.glob("data/raw/*.csv")
ball_files = [f for f in all_files if "_info" not in f]

print(f"Found {len(ball_files)} ball-by-ball CSV files.")

all_matches = []

for f in ball_files:
    if len(all_matches) >= 30:  # Stop after 30 matches
        break
    try:
        df = pd.read_csv(f)
        # If required columns are missing, try skipping first few rows (metadata in some files)
        if 'ball' not in df.columns or 'runs_off_bat' not in df.columns:
            df = pd.read_csv(f, skiprows=5)
        if 'ball' in df.columns and 'runs_off_bat' in df.columns:
            all_matches.append(df)
        else:
            print(f"Skipping file (missing columns): {f}")
    except Exception as e:
        print(f"Skipping file due to error: {f} | {e}")

if len(all_matches) == 0:
    raise Exception("No valid CSV files found for the first 30 matches!")

# 4️⃣ Combine and clean

combined_df = pd.concat(all_matches, ignore_index=True)


combined_df.columns = combined_df.columns.str.lower().str.replace(" ", "_")

keep_cols = ['match_id', 'innings', 'ball', 'batting_team', 'bowling_team',
             'striker', 'non_striker', 'bowler', 'runs_off_bat', 'extras', 'wicket_type']
combined_df = combined_df[[c for c in keep_cols if c in combined_df.columns]]

combined_df = combined_df.dropna(subset=['runs_off_bat', 'ball'])

combined_df = combined_df[combined_df['runs_off_bat'] >= 0]

os.makedirs("data/cleaned", exist_ok=True)
combined_path = "data/cleaned/combined_30_matches.csv"
combined_df.to_csv(combined_path, index=False)

print(f"✅ Combined cleaned CSV saved: {combined_path}")
print(f"Rows: {len(combined_df)}, Columns: {len(combined_df.columns)}")

