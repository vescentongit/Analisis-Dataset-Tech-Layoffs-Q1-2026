"""
Analisis Tech Layoffs Q1 2026 - Grafik 1: Analisis Multidimensi
library used : pandas, numpy, matplotlib
output       : analisis_phk_multidimensi.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from collections import Counter
import re

# ==================== LOADING AND CLEANING DATA ====================
df = pd.read_csv("data.csv")
for col in ["jobs_cut", "simultaneous_ai_investment_bn", "stock_change_day_pct",
            "pct_workforce_cut", "company_revenue_2025_bn"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# print("=== RINGKASAN DATA ===")
# print(f"Jumlah baris       : {len(df)}")
# print(f"Jumlah kolom       : {len(df.columns)}")
# print(f"Total PHK          : {df['jobs_cut'].sum():,.0f} pekerjaan")
# print(f"Perusahaan kutip AI: {df['ai_cited'].sum()} dari {len(df)}")
# print(f"Rata-rata % PHK    : {df['pct_workforce_cut'].mean():.1f}% dari total karyawan")


# ==================== PENGELOMPOKAN SEKTOR ====================
sector_summary = (
    df.groupby("sector")
    .agg(total_phk=("jobs_cut", "sum"), jumlah_perusahaan=("company", "count"),
        ai_pct=("ai_cited", "mean"),   avg_stock=("stock_change_day_pct", "mean"))
    .sort_values("total_phk", ascending=False).reset_index()
)
sector_summary["ai_pct"] = (sector_summary["ai_pct"] * 100).round(1)

# print("\n=== TOP 5 SEKTOR TERDAMPAK ===")
# print(sector_summary[["sector", "total_phk", "jumlah_perusahaan", "ai_pct"]].head(5).to_string(index=False))


# ==================== JOBS CUT BY AI ====================
correct  = df.groupby("ai_cited")["jobs_cut"].sum()
ai_group = df.groupby("ai_cited").agg(
    total_phk=("jobs_cut", "sum"), avg_pct_cut=("pct_workforce_cut", "mean"),
    avg_stock=("stock_change_day_pct", "mean"),
    avg_ai_invest=("simultaneous_ai_investment_bn", "mean"),
).reset_index()
ai_group["ai_cited"] = ai_group["ai_cited"].map({True: "AI Dikutip", False: "Non-AI"})

# print("\n=== AI vs NON-AI (berdasarkan jumlah pekerjaan) ===")
# print(ai_group.to_string(index=False))


# ==================== 4. INSIGHT STATISTIK ====================
# print("\n=== INSIGHT STATISTIK ===")

# corr_ai_phk = df[["simultaneous_ai_investment_bn", "jobs_cut"]].corr().iloc[0, 1]
# print(f"Korelasi investasi AI & jumlah PHK : {corr_ai_phk:.2f}")

# positive_big = df[(df["stock_reaction"] == "Positive") & (df["jobs_cut"] >= 5000)]
# print(f"\nPHK besar (5K+) tapi saham naik   : {len(positive_big)} perusahaan")
# print(positive_big[["company", "jobs_cut", "stock_change_day_pct"]].to_string(index=False))

# all_roles = " ".join(df["roles_most_affected"].dropna().str.lower())
# words     = re.findall(r'\b[a-z]{4,}\b', all_roles)
# stop      = {"roles", "from", "most", "that", "with", "this", "their", "have", "been"}
# common    = Counter(w for w in words if w not in stop).most_common(10)
# print("\nKata kunci peran paling terdampak PHK:")
# for word, count in common:
#     print(f"  {word:20s}: {count}x")

# country_summary = df.groupby("country")["jobs_cut"].sum().sort_values(ascending=False)
# print("\nTotal PHK per negara (top 5):")
# print(country_summary.head(5).to_string())


# ==================== GLOBAL COLOR VARIABLS ====================
BLUE   = "#1565C0"
GRAY   = "#78909C"
RED    = "#C62828"
GREEN  = "#2E7D32"
ORANGE = "#E65100"


# ==================== MULTIDIMENSIONAL ANALYSIS ====================
fig = plt.figure(figsize=(20, 16))
fig.suptitle("Analisis Dataset Tech Layoffs Q1 2026", fontsize=17, fontweight="bold", y=0.99)
gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

# 1a Horizontal bar sektor terdampak (top 10)
ax1 = fig.add_subplot(gs[0, :2])
top_sector = sector_summary.head(10)
colors_s   = [BLUE if ai > 50 else GRAY for ai in top_sector["ai_pct"]]
bars = ax1.barh(range(len(top_sector)), top_sector["total_phk"], color=colors_s, height=0.6)
ax1.set_yticks(range(len(top_sector)))
ax1.set_yticklabels(top_sector["sector"], fontsize=9)
ax1.set_xlabel("Jobs Layed Off", fontsize=9)
ax1.set_title("Total PHK per Sektor  (biru = mayoritas faktor AI)", fontsize=10, pad=8)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar in bars:
    w = bar.get_width()
    ax1.text(w + 300, bar.get_y() + bar.get_height()/2, f"{int(w):,}", va="center", fontsize=8)
ax1.set_xlim(0, top_sector["total_phk"].max() * 1.18)
ax1.spines[["top", "right"]].set_visible(False)

# 1b Pie chart proporsi PHK akibat AI vs Non-AI
ax2 = fig.add_subplot(gs[0, 2])
ax2.pie([correct[True], correct[False]], labels=["AI", "Non-AI"],
        autopct="%1.1f%%", colors=[BLUE, "#90A4AE"], startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 9})
ax2.set_title("Proporsi Penyebab Layoff\n(berdasarkan jumlah pekerjaan)", fontsize=10, pad=8)

# 1c Reaksi pasar saham terhadap pengumuman PHK
ax3 = fig.add_subplot(gs[1, 0])
stock_counts = df["stock_reaction"].value_counts()
pal     = {"Positive": GREEN, "Negative": RED, "Neutral": ORANGE}
bcolors = [pal.get(k, "#888") for k in stock_counts.index]
ax3.bar(stock_counts.index, stock_counts.values, color=bcolors, edgecolor="white", width=0.5)
ax3.set_title("Sentimen Stock Market", fontsize=10, pad=8)
ax3.set_ylabel("Jumlah Perusahaan", fontsize=9)
ax3.tick_params(axis="x", labelsize=9)
ax3.set_ylim(0, stock_counts.max() * 1.2)
ax3.spines[["top", "right"]].set_visible(False)
for i, (idx, val) in enumerate(stock_counts.items()):
    ax3.text(i, val + 0.2, str(val), ha="center", fontweight="bold", fontsize=10)

# 1d Scatter plot investasi AI vs jumlah PHK
ax4 = fig.add_subplot(gs[1, 1:])
scatter_df = df[df["simultaneous_ai_investment_bn"] > 0].copy()
sizes = np.clip(scatter_df["pct_workforce_cut"] * 15, 30, 600)
sc = ax4.scatter(scatter_df["simultaneous_ai_investment_bn"], scatter_df["jobs_cut"],
                s=sizes, c=scatter_df["stock_change_day_pct"],
                cmap="RdYlGn", alpha=0.8, edgecolors="gray", linewidths=0.5)
plt.colorbar(sc, ax=ax4, label="Perubahan Saham (%)", pad=0.02)
for _, row in scatter_df.iterrows():
    if row["jobs_cut"] >= 2000 or row["simultaneous_ai_investment_bn"] >= 5:
        ax4.annotate(row["company"].split()[0],
                    (row["simultaneous_ai_investment_bn"], row["jobs_cut"]),
                    fontsize=7.5, xytext=(5, 5), textcoords="offset points", color="#333")
ax4.set_xlabel("Investasi AI bersamaan (miliar USD)", fontsize=9)
ax4.set_ylabel("Jumlah PHK", fontsize=9)
ax4.set_title("Investasi AI vs Jumlah PHK  (size = % jobs cut)", fontsize=10, pad=8)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax4.spines[["top", "right"]].set_visible(False)

# 1e % PHK terbesar tiap perusahaan (top 10)
ax5 = fig.add_subplot(gs[2, :2])
top_pct = df.nlargest(10, "pct_workforce_cut")[["company", "pct_workforce_cut", "ai_cited"]].reset_index(drop=True)
bar_c5  = [BLUE if a else GRAY for a in top_pct["ai_cited"]]
ax5.bar(range(len(top_pct)), top_pct["pct_workforce_cut"], color=bar_c5, width=0.6)
ax5.set_xticks(range(len(top_pct)))
ax5.set_xticklabels(top_pct["company"], rotation=35, ha="right", fontsize=8)
ax5.set_ylabel("% Karyawan Di-PHK", fontsize=9)
ax5.set_title("Perusahaan dengan % PHK Terbesar  (biru = faktor AI)", fontsize=10, pad=8)
ax5.set_ylim(0, top_pct["pct_workforce_cut"].max() * 1.18)
ax5.spines[["top", "right"]].set_visible(False)
for i, val in enumerate(top_pct["pct_workforce_cut"]):
    ax5.text(i, val + 0.8, f"{val:.0f}%", ha="center", fontsize=8, fontweight="bold")

# 1f Heatmap korelasi antar variabel numerik utama
ax6 = fig.add_subplot(gs[2, 2])
num_cols  = ["jobs_cut", "pct_workforce_cut", "stock_change_day_pct",
            "simultaneous_ai_investment_bn", "company_revenue_2025_bn"]
corr      = df[num_cols].corr().round(2)
short_lbl = ["PHK", "% Cut", "Saham", "AI Inv", "Revenue"]
im = ax6.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
ax6.set_xticks(range(5)); ax6.set_yticks(range(5))
ax6.set_xticklabels(short_lbl, rotation=40, ha="right", fontsize=8)
ax6.set_yticklabels(short_lbl, fontsize=8)
for i in range(5):
    for j in range(5):
        ax6.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                fontsize=7.5, color="white" if abs(corr.values[i,j]) > 0.5 else "black")
ax6.set_title("Korelasi Antar Variabel", fontsize=10, pad=8)
plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

plt.savefig("res/analisis_phk_multidimensi.png", dpi=150, bbox_inches="tight")
plt.show()