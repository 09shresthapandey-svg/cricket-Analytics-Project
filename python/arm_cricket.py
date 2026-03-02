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
    any_boundary = g["runs_off_bat"].isin([4, 6]).any()
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


all_items = sorted({it for r in rows for it in r})
onehot = pd.DataFrame([{it: (it in r) for it in all_items} for r in rows]).astype(bool)


sample_csv = f"{OUT_DIR}/arm_transactions_sample.csv"
onehot.head(25).to_csv(sample_csv, index=False)
print(f"Saved transaction sample CSV: {sample_csv}")


sample_df = onehot.head(15).copy()
plt.figure(figsize=(12, 3))
plt.axis("off")
tbl = plt.table(cellText=sample_df.values,
                colLabels=sample_df.columns,
                cellLoc="center",
                loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.2)
plt.title("ARM Transaction Sample (One-Hot, First 15 Rows)", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/arm_transactions_sample.png", dpi=200)
plt.close()

MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.30
MIN_LIFT = 1.0

freq = apriori(onehot, min_support=MIN_SUPPORT, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules = rules[rules["lift"] > MIN_LIFT].copy()

def fmt_itemset(s):
    return ", ".join(sorted(list(s)))

rules["A"] = rules["antecedents"].apply(fmt_itemset)
rules["C"] = rules["consequents"].apply(fmt_itemset)
rules["rule"] = rules["A"] + " → " + rules["C"]

top_support = rules.sort_values("support", ascending=False).head(15)
top_conf = rules.sort_values("confidence", ascending=False).head(15)
top_lift = rules.sort_values("lift", ascending=False).head(15)

top_support.to_csv(f"{OUT_DIR}/top15_support.csv", index=False)
top_conf.to_csv(f"{OUT_DIR}/top15_confidence.csv", index=False)
top_lift.to_csv(f"{OUT_DIR}/top15_lift.csv", index=False)
print("Saved top rules CSVs in outputs/")


def plot_top15(df_top, metric, filename):
    plt.figure(figsize=(12, 7))
    y = df_top["rule"]
    x = df_top[metric]
    plt.barh(y, x)
    plt.gca().invert_yaxis()
    plt.xlabel(metric.title())
    plt.title(f"Top 15 ARM Rules by {metric.title()}")
    plt.tight_layout()
    plt.savefig(f"{OUT_IMG}/{filename}", dpi=200)
    plt.close()

plot_top15(top_support, "support", "arm_top15_support.png")
plot_top15(top_conf, "confidence", "arm_top15_confidence.png")
plot_top15(top_lift, "lift", "arm_top15_lift.png")


G = nx.DiGraph()
for _, r in top_lift.iterrows():
    a = r["A"]
    c = r["C"]
    G.add_edge(a, c, weight=float(r["lift"]))

plt.figure(figsize=(11, 7))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=900)
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edges(G, pos, arrows=True)
plt.title("ARM Network (Top Lift Rules)")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/arm_network.png", dpi=200)
plt.close()

print("✅ ARM images saved in imagesss/:")
print(" - arm_transactions_sample.png")
print(" - arm_top15_support.png")
print(" - arm_top15_confidence.png")
print(" - arm_top15_lift.png")
print(" - arm_network.png")
print(f"Thresholds used: min_support={MIN_SUPPORT}, min_confidence={MIN_CONFIDENCE}, min_lift>{MIN_LIFT}")
