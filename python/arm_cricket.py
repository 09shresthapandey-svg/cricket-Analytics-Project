import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules

CSV_PATH = "data/cleaned/combined_30_matches.csv"
OUT_IMG = "imagesss"
OUT_DIR = "outputs"
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH).dropna(subset=["ball", "runs_off_bat", "extras"])
df["over"] = df["ball"].astype(float).apply(lambda x: int(np.floor(x)))

grp = df.groupby(["match_id", "innings", "over"])

rows = []
for _, g in grp:
    total_runs = g["runs_off_bat"].sum()
    any_wicket = g["wicket_type"].notna().any()
    any_boundary = g["runs_off_bat"].isin([4,6]).any()
    any_extras = (g["extras"] > 0).any()
    dot_balls = (g["runs_off_bat"] == 0).sum()

    items = []
    if any_wicket: items.append("WICKET")
    if any_boundary: items.append("BOUNDARY")
    if any_extras: items.append("EXTRAS")
    if total_runs >= 10: items.append("HIGH_RUN_OVER")
    if dot_balls >= 3: items.append("DOT_HEAVY")

    if items:
        rows.append(items)

# One-hot
all_items = sorted({it for r in rows for it in r})
onehot = pd.DataFrame([{it: (it in r) for it in all_items} for r in rows]).astype(bool)

onehot.head(25).to_csv(f"{OUT_DIR}/arm_transactions_sample.csv", index=False)

freq = apriori(onehot, min_support=0.05, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=0.3)
rules = rules[rules["lift"] > 1.0].copy()

top_support = rules.sort_values("support", ascending=False).head(15)
top_conf = rules.sort_values("confidence", ascending=False).head(15)
top_lift = rules.sort_values("lift", ascending=False).head(15)

top_support.to_csv(f"{OUT_DIR}/top15_support.csv", index=False)
top_conf.to_csv(f"{OUT_DIR}/top15_confidence.csv", index=False)
top_lift.to_csv(f"{OUT_DIR}/top15_lift.csv", index=False)

print("Saved top rules CSVs in outputs/")

# Network plot (top lift)
G = nx.DiGraph()
for _, r in top_lift.iterrows():
    a = ", ".join(list(r["antecedents"]))
    c = ", ".join(list(r["consequents"]))
    G.add_edge(a, c, weight=float(r["lift"]))

plt.figure(figsize=(11,7))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=900)
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edges(G, pos, arrows=True)
plt.title("ARM Network (Top Lift Rules)")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/arm_network.png", dpi=200)
plt.show()
