import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Styling
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

df = pd.read_csv('../data.csv')
print(f"Shape: {df.shape}")

df.head()
df.info()
df['jobs_cut'].describe()

LEAKAGE_COLS = ['pct_workforce_cut', 'laid_off_vs_headcount_pct',
                'layoff_size_category', 'stock_change_day_pct',
                'stock_reaction']

USELESS_COLS = ['company', 'hq_city', 'ceo_quote', 'verified_source',
                'data_as_of', 'layoff_date', 'reason_stated',
                'roles_most_affected', 'replacement_roles', 'month',
                'country', 'sector']

df_clean = df.drop(columns=LEAKAGE_COLS + USELESS_COLS)
print(f"Shape setelah drop: {df_clean.shape}")
print(f"Kolom tersisa: {df_clean.columns.tolist()}")

df_clean['ai_cited'] = df_clean['ai_cited'].astype(int)
df_encoded = pd.get_dummies(df_clean, columns=['region', 'quarter'], drop_first=False)

# Konversi bool ke int biar konsisten
for col in df_encoded.columns:
    if df_encoded[col].dtype == bool:
        df_encoded[col] = df_encoded[col].astype(int)

print(f"Shape setelah encoding: {df_encoded.shape}")
df_encoded.head()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

y_raw = df_encoded['jobs_cut'].values.astype(float)
y_log = np.log1p(y_raw)

axes[0].hist(y_raw, bins=15, color='#3b82f6', edgecolor='white')
axes[0].set_title('Distribusi jobs_cut (Skala Asli)', fontweight='bold')
axes[0].set_xlabel('Jumlah PHK')
axes[0].set_ylabel('Frekuensi')

axes[1].hist(y_log, bins=15, color='#10b981', edgecolor='white')
axes[1].set_title('Distribusi jobs_cut (Skala Log)', fontweight='bold')
axes[1].set_xlabel('log(1 + jobs_cut)')
axes[1].set_ylabel('Frekuensi')

plt.tight_layout()
plt.savefig('../res_tubes/fig1_distribusi_target.png', bbox_inches='tight', dpi=150)
plt.show()

num_features = ['jobs_cut', 'company_revenue_2025_bn', 'pre_layoff_headcount',
                'simultaneous_ai_investment_bn', 'layoffs_2024', 'layoffs_2025']
corr = df[num_features].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, vmin=-1, vmax=1, linewidths=0.5, ax=ax)
ax.set_title('Matriks Korelasi: Fitur Numerik vs Target', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('../res_tubes/fig2_korelasi.png', bbox_inches='tight', dpi=150)
plt.show()

# Korelasi terurut terhadap jobs_cut
print("\nKorelasi dengan jobs_cut:")
print(corr['jobs_cut'].drop('jobs_cut').sort_values(ascending=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

region_data = df.groupby('region')['jobs_cut'].mean().sort_values(ascending=False)
axes[0].bar(region_data.index, region_data.values,
            color=['#ef4444', '#f59e0b', '#10b981'], edgecolor='white')
axes[0].set_title('Rata-rata PHK per Wilayah', fontweight='bold')
axes[0].set_ylabel('Rata-rata jobs_cut')
axes[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(region_data.values):
    axes[0].text(i, v + 100, f'{v:,.0f}', ha='center', fontweight='bold')

ai_data = df.groupby('ai_cited')['jobs_cut'].mean()
axes[1].bar(['AI tidak disebut', 'AI disebut'], ai_data.values,
            color=['#6b7280', '#3b82f6'], edgecolor='white')
axes[1].set_title('Rata-rata PHK: AI Cited vs Tidak', fontweight='bold')
axes[1].set_ylabel('Rata-rata jobs_cut')
for i, v in enumerate(ai_data.values):
    axes[1].text(i, v + 100, f'{v:,.0f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../res_tubes/fig3_kategori.png', bbox_inches='tight', dpi=150)
plt.show()

X = df_encoded.drop(columns=['jobs_cut']).values.astype(float)
y_raw = df_encoded['jobs_cut'].values.astype(float)
y_log = np.log1p(y_raw)

feature_names = [c for c in df_encoded.columns if c != 'jobs_cut']
print(f"Jumlah fitur: {len(feature_names)}")
print(f"Fitur: {feature_names}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"X_scaled shape: {X_scaled.shape}")
print(f"Mean tiap fitur (sudah ~0): {X_scaled.mean(axis=0).round(3)}")
print(f"Std tiap fitur (sudah ~1):  {X_scaled.std(axis=0).round(3)}")

models = {
    'KNN (K=5)': KNeighborsRegressor(n_neighbors=5),
    'KNN (K=7)': KNeighborsRegressor(n_neighbors=7),
    'Ridge Regression': Ridge(alpha=10.0),
    'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=3,
                                        random_state=RANDOM_SEED)
}

loo = LeaveOneOut()
results = []
predictions_log = {}

for name, model in models.items():
    # Prediksi di skala log
    y_pred_log = cross_val_predict(model, X_scaled, y_log, cv=loo)
    # Transformasi balik ke skala asli
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)

    r2_orig = r2_score(y_raw, y_pred)
    r2_log = r2_score(y_log, y_pred_log)
    mae = mean_absolute_error(y_raw, y_pred)
    rmse = np.sqrt(mean_squared_error(y_raw, y_pred))
    mape = np.mean(np.abs((y_raw - y_pred) / y_raw)) * 100

    results.append({
        'Model': name,
        'R² (original)': round(r2_orig, 4),
        'R² (log-scale)': round(r2_log, 4),
        'MAE': round(mae, 0),
        'RMSE': round(rmse, 0),
        'MAPE (%)': round(mape, 2)
    })
    predictions_log[name] = y_pred_log

results_df = pd.DataFrame(results).sort_values('R² (log-scale)', ascending=False).reset_index(drop=True)
results_df

best_model_name = results_df.iloc[0]['Model']
best_r2_log = results_df.iloc[0]['R² (log-scale)']
best_mae = results_df.iloc[0]['MAE']
print(f"Model terbaik: {best_model_name}")
print(f"R² (log-scale): {best_r2_log}")
print(f"MAE: {best_mae:,.0f} PHK")

fig, ax = plt.subplots(figsize=(9, 4.5))
colors = ['#10b981' if m == best_model_name else '#94a3b8' for m in results_df['Model']]
bars = ax.barh(results_df['Model'], results_df['R² (log-scale)'],
            color=colors, edgecolor='white')
ax.set_xlabel('R² (Leave-One-Out CV, log-scale)')
ax.set_title('Perbandingan Performa Model', fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
for bar, val in zip(bars, results_df['R² (log-scale)']):
    ax.text(val + 0.005 if val >= 0 else val - 0.005,
            bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center',
            ha='left' if val >= 0 else 'right', fontweight='bold')
plt.tight_layout()
plt.savefig('../res_tubes/fig4_model_comparison.png', bbox_inches='tight', dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(7, 6))
y_pred_best_log = predictions_log[best_model_name]
ax.scatter(y_log, y_pred_best_log, alpha=0.7, s=90, color='#10b981',
        edgecolor='black', linewidth=0.8)
mn, mx = min(y_log.min(), y_pred_best_log.min()), max(y_log.max(), y_pred_best_log.max())
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Garis Ideal')
ax.set_xlabel('log(1 + jobs_cut) Aktual', fontweight='bold')
ax.set_ylabel('log(1 + jobs_cut) Prediksi', fontweight='bold')
ax.set_title(f'Prediksi vs Aktual — {best_model_name}\n'
            f'R² (log-scale) = {best_r2_log:.4f} | MAE = {best_mae:,.0f}',
            fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('../res_tubes/fig5_pred_vs_actual.png', bbox_inches='tight', dpi=150)
plt.show()

rf_final = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=RANDOM_SEED)
rf_final.fit(X_scaled, y_log)

importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_final.feature_importances_
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(importance['Feature'], importance['Importance'],
        color='#8b5cf6', edgecolor='white')
ax.set_xlabel('Feature Importance')
ax.set_title('Feature Importance (Random Forest)', fontweight='bold')
for i, v in enumerate(importance['Importance']):
    ax.text(v + 0.003, i, f'{v:.3f}', va='center')
plt.tight_layout()
plt.savefig('../res_tubes/fig6_feature_importance.png', bbox_inches='tight', dpi=150)
plt.show()

print('\nTop 5 fitur paling berpengaruh:')
print(importance.sort_values('Importance', ascending=False).head().to_string(index=False))

def predict_jobs_cut(ai_cited, revenue_bn, headcount, ai_invest_bn,
                    layoffs_2024, layoffs_2025, region, quarter='Q1 2026'):
    """
    Prediksi jumlah PHK berdasarkan karakteristik perusahaan.

    Parameter:
        ai_cited: 0 atau 1 (apakah AI disebut sebagai alasan)
        revenue_bn: revenue perusahaan dalam miliar USD
        headcount: jumlah karyawan sebelum PHK
        ai_invest_bn: simultaneous AI investment dalam miliar USD
        layoffs_2024: jumlah PHK perusahaan ini di 2024
        layoffs_2025: jumlah PHK perusahaan ini di 2025
        region: 'North America', 'Europe', atau 'Asia-Pacific'
        quarter: default 'Q1 2026'
    """
    # Build feature vector sesuai urutan training
    row = {
        'ai_cited': ai_cited,
        'company_revenue_2025_bn': revenue_bn,
        'pre_layoff_headcount': headcount,
        'simultaneous_ai_investment_bn': ai_invest_bn,
        'layoffs_2024': layoffs_2024,
        'layoffs_2025': layoffs_2025,
        'region_Asia-Pacific': 1 if region == 'Asia-Pacific' else 0,
        'region_Europe': 1 if region == 'Europe' else 0,
        'region_North America': 1 if region == 'North America' else 0,
        'quarter_Q1 2026': 1 if quarter == 'Q1 2026' else 0,
    }
    x = np.array([[row[f] for f in feature_names]], dtype=float)
    x_scaled = scaler.transform(x)
    y_pred_log = rf_final.predict(x_scaled)[0]
    y_pred = np.expm1(y_pred_log)
    return int(round(y_pred))

# Contoh: prediksi PHK untuk perusahaan hipotetis
example = predict_jobs_cut(
    ai_cited=1,
    revenue_bn=50.0,
    headcount=20000,
    ai_invest_bn=5.0,
    layoffs_2024=500,
    layoffs_2025=1000,
    region='North America'
)
print(f'Estimasi PHK: {example:,} karyawan')