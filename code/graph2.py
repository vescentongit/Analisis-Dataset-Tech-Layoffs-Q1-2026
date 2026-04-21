"""
Analisis Dataset Tech Layoffs Q1 2026 - Time-Series
library used : pandas, numpy, matplotlib
output       : analisis_phk_timeseries.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# ==================== LOADING & CLEANING DATA ====================
df = pd.read_csv("data.csv")
for col in ["jobs_cut", "simultaneous_ai_investment_bn", "stock_change_day_pct",
            "pct_workforce_cut", "company_revenue_2025_bn"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["layoff_date"]  = pd.to_datetime(df["layoff_date"])
df["week"]         = df["layoff_date"].dt.to_period("W").apply(lambda r: r.start_time)
df["day_of_month"] = df["layoff_date"].dt.day
df["month_num"]    = df["layoff_date"].dt.month

print(f"Rentang tanggal : {df['layoff_date'].min().date()} s/d {df['layoff_date'].max().date()}")


# ==================== PENGELOMPOKAN DATA BASED ON TIME SERIES ====================
monthly = (
    df.groupby("month_num").agg(
        total_phk=("jobs_cut", "sum"),
        jumlah_event=("company", "count"),
        avg_pct_cut=("pct_workforce_cut", "mean"),
        avg_stock=("stock_change_day_pct", "mean"),
        ai_phk=("jobs_cut", lambda x: x[df.loc[x.index, "ai_cited"]].sum()),
        nonai_phk=("jobs_cut", lambda x: x[df.loc[x.index, "ai_cited"]].sum()),
    ).reset_index()
)
monthly["month_label"]    = monthly["month_num"].map({1: "Januari", 2: "Februari", 3: "Maret"})
monthly["ai_ratio"]       = (monthly["ai_phk"] / monthly["total_phk"] * 100).round(1)
monthly["cumulative_phk"] = monthly["total_phk"].cumsum()

weekly = (
    df.groupby("week").agg(
        total_phk=("jobs_cut", "sum"),
        jumlah_event=("company", "count"),
    ).reset_index()
)
weekly["cumulative_phk"] = weekly["total_phk"].cumsum()

timeline_df = df.sort_values("layoff_date")[["company", "layoff_date", "jobs_cut", "ai_cited", "sector"]].copy()

early = df[df["day_of_month"] <= 10]["jobs_cut"].sum()
mid   = df[(df["day_of_month"] > 10) & (df["day_of_month"] <= 20)]["jobs_cut"].sum()
late  = df[df["day_of_month"] > 20]["jobs_cut"].sum()
total_all = early + mid + late


# ==================== SUMMARY (READ CLI) ====================
# print("\n=== TIME-SERIES: RINGKASAN PER BULAN ===")
# print(monthly[["month_label", "total_phk", "jumlah_event", "ai_phk", "nonai_phk", "ai_ratio", "avg_stock"]].to_string(index=False))

# print("\n=== TIME-SERIES: PERCEPATAN PHK ===")
# for i in range(1, len(monthly)):
#     prev = monthly.loc[i-1, "total_phk"]
#     curr = monthly.loc[i,   "total_phk"]
#     print(f"  {monthly.loc[i-1,'month_label']} → {monthly.loc[i,'month_label']}: {(curr-prev)/prev*100:+.1f}% perubahan volume")

# print("\n=== TIME-SERIES: POLA HARI PENGUMUMAN ===")
# print(f"  Awal bulan  (1–10) : {early:,} PHK ({early/total_all*100:.1f}%)")
# print(f"  Tengah bulan(11–20): {mid:,} PHK ({mid/total_all*100:.1f}%)")
# print(f"  Akhir bulan (21–31): {late:,} PHK ({late/total_all*100:.1f}%)")

# print("\n=== INSIGHT TIME-SERIES ===")
# peak_month = monthly.loc[monthly["total_phk"].idxmax(), "month_label"]
# peak_vol   = monthly["total_phk"].max()
# print(f"[1] Bulan dengan PHK tertinggi : {peak_month} ({peak_vol:,} pekerjaan)")

# fastest_week = weekly.loc[weekly["total_phk"].idxmax()]
# print(f"[2] Minggu paling aktif        : {fastest_week['week'].date()} ({int(fastest_week['total_phk']):,} PHK)")

# print(f"[3] {early/total_all*100:.0f}% PHK diumumkan di awal bulan (1–10) — pola 'awal kuartal'")

# jan_ai = monthly.loc[monthly["month_num"] == 1, "ai_ratio"].values[0]
# mar_ai = monthly.loc[monthly["month_num"] == 3, "ai_ratio"].values[0]
# print(f"[4] Proporsi PHK karena AI: {jan_ai:.0f}% (Jan) → {mar_ai:.0f}% (Mar)")
# print(f"    → Tren AI-driven layoffs makin dominan menjelang akhir kuartal")

# avg_gap = df.groupby("month_num")["jobs_cut"].mean()
# print(f"[5] Rata-rata PHK per event: Jan={avg_gap[1]:,.0f} | Feb={avg_gap[2]:,.0f} | Mar={avg_gap[3]:,.0f}")
# print(f"    → Skala per event menurun di Maret, tapi frekuensi event tetap tinggi")


# ==================== GLOBAL VARIABLES ====================
BLUE   = "#1565C0"
GRAY   = "#78909C"
RED    = "#C62828"
GREEN  = "#2E7D32"
ORANGE = "#E65100"
BLACK  = "#000000"
GRAY   = "#78909C"

# ==================== TIME-BASED ANALYSIS & VISUALIZATION ====================
fig2, axes = plt.subplots(3, 2, figsize=(18, 17))
fig2.suptitle("Analisis Dataset Tech Layoffs Q1 2026 - Time-Series",
            fontsize=15, fontweight="bold", y=0.995)
plt.subplots_adjust(hspace=0.6, wspace=0.38)

# 2a Bar grouped AI/Non-AI per bulan + overlay saham
ax = axes[0, 0]
x  = np.arange(len(monthly)); w = 0.35
b1 = ax.bar(x - w/2, monthly["ai_phk"],    width=w, label="AI dikutip", color=BLUE, alpha=0.85)
b2 = ax.bar(x + w/2, monthly["nonai_phk"], width=w, label="Non-AI",     color=GRAY, alpha=0.85)
ax2b = ax.twinx()
ax2b.plot(x, monthly["avg_stock"], color=GREEN, marker="o", linewidth=2)
ax2b.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax2b.set_ylabel("Avg perubahan saham (%)", color=GREEN, fontsize=8)
ax2b.tick_params(axis="y", labelcolor=GREEN, labelsize=8)
ax2b.set_ylim(-2, monthly["avg_stock"].max() * 2.2)
ax.set_xticks(x); ax.set_xticklabels(monthly["month_label"], fontsize=9)
ax.set_ylabel("Jumlah PHK", fontsize=9)
ax.set_title("PHK per Bulan\n+\nSentimen Stock Market", fontsize=10, pad=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_ylim(0, max(monthly["ai_phk"].max(), monthly["nonai_phk"].max()) * 1.35)
ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
ax.spines[["top", "right"]].set_visible(False)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    if h > 500:
        ax.text(bar.get_x() + bar.get_width()/2, h + 600, f"{int(h):,}",
                ha="center", fontsize=7, fontweight="bold")

# 2b Jumlah kumulatif PHK per minggu
ax = axes[0, 1]
ax.fill_between(weekly["week"], weekly["cumulative_phk"], alpha=0.15, color=BLUE)
ax.plot(weekly["week"], weekly["cumulative_phk"], color=BLUE, linewidth=2.5, marker="o", markersize=5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_ylabel("Total PHK kumulatif", fontsize=9)
ax.set_title("Akumulasi PHK per Minggu\n(kurva percepatan)", fontsize=10, pad=8)
ax.spines[["top", "right"]].set_visible(False)
for _, row in weekly.iterrows():
    if row["total_phk"] >= 10000:
        ax.annotate(f"+{int(row['total_phk']):,}",
                    xy=(row["week"], row["cumulative_phk"]),
                    xytext=(8, -14), textcoords="offset points",
                    fontsize=8, color=BLUE, fontweight="bold")

# 2c Scatter timeline event
ax = axes[1, 0]
for _, row in timeline_df.iterrows():
    color = BLUE if row["ai_cited"] else GRAY
    size  = np.clip(row["jobs_cut"] / 50, 20, 800)
    ax.scatter(row["layoff_date"], row["jobs_cut"], s=size, color=color,
            alpha=0.75, edgecolors="white", linewidths=0.5)
    if row["jobs_cut"] >= 5000:
        ax.annotate(row["company"].split()[0],
                    (row["layoff_date"], row["jobs_cut"]),
                    xytext=(5, 6), textcoords="offset points", fontsize=8, color="#222")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_ylabel("Jumlah PHK", fontsize=9)
ax.set_title("Timeline Event PHK\n(size = volume)", fontsize=10, pad=8)
ax.spines[["top", "right"]].set_visible(False)
legend_el = [Line2D([0],[0], marker="o", color="w", markerfacecolor=BLUE, markersize=8, label="AI"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor=GRAY, markersize=8, label="Non-AI")]
ax.legend(handles=legend_el, fontsize=8, loc="upper left")

# 2d Stacked 100% komposisi AI per bulan
ax = axes[1, 1]
x  = np.arange(len(monthly))
ax.bar(x, monthly["ai_ratio"],                label="AI dikutip", color=BLUE, alpha=0.85)
ax.bar(x, 100 - monthly["ai_ratio"], bottom=monthly["ai_ratio"], label="Non-AI", color=GRAY, alpha=0.85)
ax.axhline(50, color="white", linewidth=1.5, linestyle="--")
ax.set_xticks(x); ax.set_xticklabels(monthly["month_label"], fontsize=9)
ax.set_ylabel("% dari total PHK bulan itu", fontsize=9)
ax.set_ylim(0, 112)
ax.set_title("Komposisi AI vs Non-AI per Bulan", fontsize=10, pad=8)
ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
ax.spines[["top", "right"]].set_visible(False)
for i, (ai_r, total) in enumerate(zip(monthly["ai_ratio"], monthly["total_phk"])):
    ax.text(i, ai_r / 2,              f"{ai_r:.0f}%",      ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.text(i, ai_r + (100-ai_r)/2,   f"{100-ai_r:.0f}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.text(i, 103, f"n={int(total):,}", ha="center", fontsize=8, color="#444")

# 2e Histogram distribusi hari pengumuman PHK dalam bulan
ax = axes[2, 0]

bins = range(1, 32)  # hari 1-31

# plot histogram frekuensi event (jumlah perusahaan per hari)
ax.hist(df["day_of_month"], bins=bins, align="left", 
        color=ORANGE, edgecolor="white", alpha=0.85, label="Frekuensi event")

# overlay: total jobs_cut per hari sebagai line chart
jobs_per_day = df.groupby("day_of_month")["jobs_cut"].sum()
ax2h = ax.twinx()
ax2h.plot(jobs_per_day.index, jobs_per_day.values,color=BLUE, 
            linewidth=2, marker="o", markersize=5, label="Total PHK")
ax2h.set_ylabel("Total jobs cut", color=BLUE, fontsize=8)
ax2h.tick_params(axis="y", labelcolor=BLUE, labelsize=8)
ax2h.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax2h.set_ylim(0, jobs_per_day.max() * 2.2)

# shading awal / tengah / akhir
ax.axvspan(1,  10, alpha=0.06, color=BLUE)
ax.axvspan(11, 20, alpha=0.06, color=ORANGE)
ax.axvspan(21, 31, alpha=0.06, color=GREEN)
ax.text(5.5,  ax.get_ylim()[1] * 0.88, "Awal",   ha="center", fontsize=7, color=BLUE)
ax.text(15.5, ax.get_ylim()[1] * 0.88, "Tengah", ha="center", fontsize=7, color=ORANGE)
ax.text(26,   ax.get_ylim()[1] * 0.88, "Akhir",  ha="center", fontsize=7, color=GREEN)

ax.set_xlabel("Tanggal dalam bulan", fontsize=9)
ax.set_ylabel("Frekuensi event PHK", fontsize=9)
ax.set_title("Histogram Hari Pengumuman PHK\n(dalam bulan, semua bulan digabung)", fontsize=10, pad=8)
ax.set_xticks([1, 5, 10, 15, 20, 25, 31])
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# 2f Frekuensi event + avg % jobs cut per bulan
ax = axes[2, 1]
x  = np.arange(len(monthly))
bars_e = ax.bar(x, monthly["jumlah_event"], color=ORANGE, alpha=0.85, width=0.5)
ax2e = ax.twinx()
ax2e.plot(x, monthly["avg_pct_cut"], color=BLUE, marker="o", linewidth=2.5, markersize=7)
ax2e.set_ylabel("Rata-rata % layoff", color=BLUE, fontsize=8)
ax2e.tick_params(axis="y", labelcolor=BLUE, labelsize=8)
ax2e.set_ylim(0, monthly["avg_pct_cut"].max() * 2)
ax.set_xticks(x); ax.set_xticklabels(monthly["month_label"], fontsize=9)
ax.set_ylabel("Jumlah event PHK", fontsize=9)
ax.set_title("Frekuensi Event & Intensitas PHK per Bulan\n(intensitas = % layoff)", fontsize=10, pad=8)
ax.set_ylim(0, monthly["jumlah_event"].max() * 1.4)
ax.spines[["top", "right"]].set_visible(False)
for bar in bars_e:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, str(int(h)),
            ha="center", fontsize=10, fontweight="bold")

plt.savefig("res/analisis_phk_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()