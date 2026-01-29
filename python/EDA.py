import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


df = pd.read_csv("data/cleaned/combined_30_matches.csv")

os.makedirs("images", exist_ok=True)


print("First 5 rows of data:")
print(df.head())
print("\nSummary of numeric columns:")
print(df.describe())


#  Distribution of runs_off_bat
plt.figure(figsize=(8,5))
sns.histplot(df['runs_off_bat'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Runs Scored per Ball')
plt.xlabel('Runs')
plt.ylabel('Frequency')
plt.savefig("images/runs_distribution.png")
plt.close()

# Number of wickets per match
wickets_per_match = df.dropna(subset=['wicket_type']).groupby('match_id').size()
plt.figure(figsize=(8,5))
sns.barplot(x=wickets_per_match.index, y=wickets_per_match.values, palette="magma")
plt.xticks(rotation=90)
plt.title('Wickets per Match')
plt.xlabel('Match ID')
plt.ylabel('Number of Wickets')
plt.savefig("images/wickets_per_match.png")
plt.close()

# Top 10 batsmen by total runs
batsmen_runs = df.groupby('striker')['runs_off_bat'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(8,5))
sns.barplot(x=batsmen_runs.values, y=batsmen_runs.index, palette="viridis")
plt.title('Top 10 Batsmen by Total Runs')
plt.xlabel('Total Runs')
plt.ylabel('Batsman')
plt.savefig("images/top_batsmen.png")
plt.close()

# Top 10 bowlers by wickets
wickets = df.dropna(subset=['wicket_type']).groupby('bowler').size().sort_values(ascending=False).head(10)
plt.figure(figsize=(8,5))
sns.barplot(x=wickets.values, y=wickets.index, palette="plasma")
plt.title('Top 10 Bowlers by Wickets')
plt.xlabel('Wickets Taken')
plt.ylabel('Bowler')
plt.savefig("images/top_bowlers.png")
plt.close()

# Team-wise total runs
team_runs = df.groupby('batting_team')['runs_off_bat'].sum().sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=team_runs.index, y=team_runs.values, palette="coolwarm")
plt.title('Total Runs by Team')
plt.xlabel('Team')
plt.ylabel('Runs')
plt.xticks(rotation=45)
plt.savefig("images/team_runs.png")
plt.close()

# Balls vs runs scatter (strike rate like)
plt.figure(figsize=(8,5))
sample_df = df.groupby('striker').agg({'ball':'count', 'runs_off_bat':'sum'})
sns.scatterplot(x=sample_df['ball'], y=sample_df['runs_off_bat'])
plt.title('Balls Faced vs Runs Scored per Player')
plt.xlabel('Balls Faced')
plt.ylabel('Total Runs')
plt.savefig("images/balls_vs_runs.png")
plt.close()

#  Extras distribution
plt.figure(figsize=(8,5))
sns.histplot(df['extras'], bins=10, kde=False, color='orange')
plt.title('Distribution of Extras per Ball')
plt.xlabel('Extras')
plt.ylabel('Frequency')
plt.savefig("images/extras_distribution.png")
plt.close()

#  Runs per innings
runs_per_innings = df.groupby('innings')['runs_off_bat'].sum()
plt.figure(figsize=(8,5))
sns.barplot(x=runs_per_innings.index, y=runs_per_innings.values, palette="mako")
plt.title('Total Runs per Innings')
plt.xlabel('Innings')
plt.ylabel('Runs')
plt.savefig("images/runs_per_innings.png")
plt.close()

# Correlation heatmap of numeric features
numeric_cols = df.select_dtypes(include='number')
plt.figure(figsize=(8,6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Features')
plt.savefig("images/correlation_heatmap.png")
plt.close()

# Wickets by type (bar chart)
wicket_counts = df['wicket_type'].value_counts()
plt.figure(figsize=(8,5))
sns.barplot(x=wicket_counts.index, y=wicket_counts.values, palette="spring")
plt.title('Wicket Types Distribution')
plt.xlabel('Wicket Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig("images/wicket_types.png")
plt.close()

print(" 10 visualizations saved in 'images/' folder.")
