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

# Load cleaned cricket data
df = pd.read_csv(CSV_PATH).dropna(subset=["ball", "runs_off_bat", "extras"])
df["over"] = df["ball"].astype(float).apply(lambda x: int(np.floor(x)))

# Group each over as one transaction
grp = df.groupby(["match_id", "innings", "over"])

rows = []
raw_transaction_rows = []

for (match_id, innings, over), g in grp:
    total_runs = g["runs_off_bat"].sum()
    any_wicket = g["wicket_type"].notna().any()
    any_boundary = g["runs_off_bat"].isin([4, 6]).any()
    any_extras = (g["extras"] > 0).any()
    dot_balls = (g["runs_off_bat"] == 0).sum()

    items = []
    if any_wicket:
        items.append("WICKET")
    if any_boundary:
        items.append("BOUNDARY")
    if any_extras:
        items.append("EXTRAS")
    if total_runs >= 10:
        items.append("HIGH_RUN_OVER")
    if dot_balls >= 3:
        items.append("DOT_HEAVY")

    if items:
        rows.append(items)
        raw_transaction_rows.append({
            "match_id": match_id,
            "innings": innings,
            "over": over,
            "items": ", ".join(items)
        })

# Save raw transactions before one-hot encoding
raw_transactions_df = pd.DataFrame(raw_transaction_rows)
raw_csv = f"{OUT_DIR}/arm_raw_transactions_sample.csv"
raw_transactions_df.head(25).to_csv(raw_csv, index=False)
print(f"Saved raw transaction sample CSV: {raw_csv}")

# Image of raw transactions
raw_sample = raw_transactions_df.head(15).copy()
plt.figure(figsize=(12, 4))
plt.axis("off")
tbl = plt.table(
    cellText=raw_sample.values,
    colLabels=raw_sample.columns,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.2)
plt.title("ARM Raw Transaction Sample (Before One-Hot Encoding)", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/arm_raw_transactions_sample.png", dpi=200, bbox_inches="tight")
plt.close()

# One-hot encode transactions
all_items = sorted({it for r in rows for it in r})
onehot = pd.DataFrame([{it: (it in r) for it in all_items} for r in rows]).astype(bool)

sample_csv = f"{OUT_DIR}/arm_transactions_sample.csv"
onehot.head(25).to_csv(sample_csv, index=False)
print(f"Saved one-hot transaction sample CSV: {sample_csv}")

# Image of one-hot transaction sample
sample_df = onehot.head(15).copy()
plt.figure(figsize=(12, 3))
plt.axis("off")
tbl = plt.table(
    cellText=sample_df.values,
    colLabels=sample_df.columns,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.2)
plt.title("ARM Transaction Sample (After One-Hot Encoding)", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/arm_transactions_sample.png", dpi=200, bbox_inches="tight")
plt.close()

# ARM metrics overview image
metrics_df = pd.DataFrame({
    "Metric": ["Support", "Confidence", "Lift"],
    "Meaning": [
        "How often a rule appears in all transactions",
        "How often the consequent occurs when the antecedent is present",
        "How much stronger the rule is than random chance"
    ]
})

plt.figure(figsize=(11, 2.8))
plt.axis("off")
tbl = plt.table(
    cellText=metrics_df.values,
    colLabels=metrics_df.columns,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.4)
plt.title("ARM Metrics Overview", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/arm_metrics_overview.png", dpi=200, bbox_inches="tight")
plt.close()

# Apriori flow image
apriori_steps = pd.DataFrame({
    "Step": [1, 2, 3, 4],
    "Apriori Process": [
        "Prepare transaction data",
        "Find frequent itemsets above min_support",
        "Generate rules from frequent itemsets",
        "Filter rules using confidence and lift"
    ]
})

plt.figure(figsize=(11, 3))
plt.axis("off")
tbl = plt.table(
    cellText=apriori_steps.values,
    colLabels=apriori_steps.columns,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.4)
plt.title("Apriori Algorithm Flow", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/arm_apriori_flow.png", dpi=200, bbox_inches="tight")
plt.close()

# ARM thresholds
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.30
MIN_LIFT = 1.0

# Apriori and rule generation
freq = apriori(onehot, min_support=MIN_SUPPORT, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules = rules[rules["lift"] > MIN_LIFT].copy()

def fmt_itemset(s):
    return ", ".join(sorted(list(s)))

rules["A"] = rules["antecedents"].apply(fmt_itemset)
rules["C"] = rules["consequents"].apply(fmt_itemset)
rules["rule"] = rules["A"] + " → " + rules["C"]

# Top 15 by each metric
top_support = rules.sort_values("support", ascending=False).head(15)
top_conf = rules.sort_values("confidence", ascending=False).head(15)
top_lift = rules.sort_values("lift", ascending=False).head(15)

top_support.to_csv(f"{OUT_DIR}/top15_support.csv", index=False)
top_conf.to_csv(f"{OUT_DIR}/top15_confidence.csv", index=False)
top_lift.to_csv(f"{OUT_DIR}/top15_lift.csv", index=False)
print("Saved top ARM rules CSVs in outputs/")

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

# Network visualization from top lift rules
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

# Save summary results
summary_df = pd.DataFrame({
    "metric": [
        "Minimum Support",
        "Minimum Confidence",
        "Minimum Lift",
        "Number of Frequent Itemsets",
        "Number of Rules After Filtering"
    ],
    "value": [
        MIN_SUPPORT,
        MIN_CONFIDENCE,
        MIN_LIFT,
        len(freq),
        len(rules)
    ]
})
summary_df.to_csv(f"{OUT_DIR}/arm_summary_results.csv", index=False)

print("ARM images saved in imagesss/:")
print(" - arm_metrics_overview.png")
print(" - arm_apriori_flow.png")
print(" - arm_raw_transactions_sample.png")
print(" - arm_transactions_sample.png")
print(" - arm_top15_support.png")
print(" - arm_top15_confidence.png")
print(" - arm_top15_lift.png")
print(" - arm_network.png")

print("ARM outputs saved in outputs/:")
print(" - arm_raw_transactions_sample.csv")
print(" - arm_transactions_sample.csv")
print(" - top15_support.csv")
print(" - top15_confidence.csv")
print(" - top15_lift.csv")
print(" - arm_summary_results.csv")

print(f"Thresholds used: min_support={MIN_SUPPORT}, min_confidence={MIN_CONFIDENCE}, min_lift>{MIN_LIFT}")
