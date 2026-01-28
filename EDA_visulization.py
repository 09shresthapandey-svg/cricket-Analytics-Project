import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/cleaned/cleaned_sample_match.csv")

# 1. Runs per ball
plt.hist(df['runs_off_bat'], bins=10)
plt.title("Distribution of Runs per Ball")
plt.xlabel("Runs")
plt.ylabel("Frequency")
plt.savefig("figures/runs_distribution.png")
plt.clf()

# 2. Extras distribution
plt.hist(df['extras'], bins=10)
plt.title("Extras Distribution")
plt.xlabel("Extras")
plt.ylabel("Frequency")
plt.savefig("figures/extras_distribution.png")
plt.clf()

# 3. Wickets by over
df.groupby("over")["is_wicket"].sum().plot(kind="bar")
plt.title("Wickets by Over")
plt.xlabel("Over")
plt.ylabel("Wickets")
plt.savefig("figures/wickets_by_over.png")
plt.clf()

# 4–10 Similar plots (strike rate proxy, venue runs, etc.)
