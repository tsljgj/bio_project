import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# ---------- LOAD DATA ----------
df = pd.read_csv("final_output/healing_results.csv")

# ---------- MAP WELL → COLUMN ----------
def extract_col(well):
    m = re.search(r'(\d+)$', well)
    return int(m.group(1)) if m else None

df["col"] = df["well"].apply(extract_col)

# ---------- MAP COLUMN → GROUP ----------
def map_group(col):
    if col == 1:
        return "control_global"
    elif col == 2:
        return "control_R"
    elif col in [3,4,5,6]:
        return "R"
    elif col == 8:
        return "control_UV"
    elif col in [9,10,11,12]:
        return "UV"
    else:
        return "ignore"

# ---------- MAP COLUMN → TIME ----------
def map_time(col):
    mapping = {
        3: "10s", 9: "10s",
        4: "1m", 10: "1m",
        5: "5m", 11: "5m",
        6: "10m", 12: "10m"
    }
    return mapping.get(col, "control")

df["group"] = df["col"].apply(map_group)
df["time"] = df["col"].apply(map_time)

# ---------- CLEAN DATA ----------
df = df[df["group"] != "ignore"]
df_plot = df[df["percent_closure"] >= 0]

TIME_ORDER = ["10s","1m","5m","10m"]

# ---------- SUMMARY ----------
summary = df_plot.groupby(["group","time"]).agg(
    mean=("percent_closure","mean"),
    sem=("percent_closure","sem")
).reset_index()

summary["time"] = pd.Categorical(summary["time"], categories=TIME_ORDER, ordered=True)

# ---------- CLEAN LINE PLOT ----------
plt.figure(figsize=(8,5))

for group, color in [("R","red"), ("UV","blue")]:
    sub = summary[summary["group"] == group].sort_values("time")
    if len(sub) == 0:
        continue

    x = np.arange(len(sub))

    plt.plot(x, sub["mean"], marker="o", linewidth=3, label=group, color=color)

    plt.fill_between(
        x,
        sub["mean"] - sub["sem"].fillna(0),
        sub["mean"] + sub["sem"].fillna(0),
        alpha=0.2,
        color=color
    )

plt.xticks(np.arange(len(TIME_ORDER)), TIME_ORDER)
plt.xlabel("Exposure Duration")
plt.ylabel("Percent Closure (24h)")
plt.title("Healing vs Exposure Time")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("final_output/clean_line_plot.png", dpi=200)
plt.close()

# ---------- BAR PLOT ----------
plt.figure(figsize=(10,5))

labels = []
means = []
errors = []
colors = []

for group, color in [("R","red"), ("UV","blue")]:
    sub = summary[summary["group"] == group].sort_values("time")
    for _, row in sub.iterrows():
        labels.append(f"{group}\n{row['time']}")
        means.append(row["mean"])
        errors.append(row["sem"] if not np.isnan(row["sem"]) else 0)
        colors.append(color)

x = np.arange(len(labels))

plt.bar(x, means, yerr=errors, capsize=5, color=colors)

plt.xticks(x, labels)
plt.ylabel("Percent Closure")
plt.title("Healing by Treatment and Time")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("final_output/bar_plot.png", dpi=200)
plt.close()

# ---------- OVERALL BAR ----------
plt.figure(figsize=(6,5))

avg = df_plot.groupby("group")["percent_closure"].mean()

groups = ["control_global","control_R","control_UV","R","UV"]
groups = [g for g in groups if g in avg.index]

vals = [avg[g] for g in groups]

plt.bar(groups, vals)
plt.ylabel("Average Percent Closure")
plt.title("Overall Healing by Treatment")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("final_output/overall_bar.png", dpi=200)
plt.close()

print("Done. Graphs saved in final_output/")