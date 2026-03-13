## ============================================================
## Lecture 2 – Data Management: Joining Tables & Time Series
## Datasets: Bitcoin (BTC) and NVIDIA (NVDA) daily price data
## ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
import os

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------

_dir = os.path.dirname(os.path.abspath(__file__))
btc = pd.read_csv(os.path.join(_dir, "Bitcoin.csv"))
nvda = pd.read_csv(os.path.join(_dir, "NVIDIA.csv"))

print("BTC shape:", btc.shape)
print("NVDA shape:", nvda.shape)

print("\nBTC head:")
print(btc.head())

print("\nNVDA head:")
print(nvda.head())

# ---------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------

# ------ 2a. Strip whitespace from column names ------
btc.columns  = btc.columns.str.strip()
nvda.columns = nvda.columns.str.strip()

# ------ 2b. Parse 'Date' column to datetime ------
btc["Date"]  = pd.to_datetime(btc["Date"],  format="%m/%d/%Y")
nvda["Date"] = pd.to_datetime(nvda["Date"], format="%m/%d/%Y")

print("\nBTC dtypes after date parse:")
print(btc.dtypes)

# ------ 2c. Clean numeric columns ------
# 'Price', 'Open', 'High', 'Low' contain commas (e.g. "88,126.0") → strip them
for col in ["Price", "Open", "High", "Low"]:
    btc[col]  = btc[col].astype(str).str.replace(",", "").astype(float)
    nvda[col] = nvda[col].astype(str).str.replace(",", "").astype(float)

# 'Vol.' uses suffixes like 67.08K or 142.75M → convert to plain numbers
def parse_volume(val):
    val = str(val).strip().replace(",", "")
    if val.endswith("K"):
        return float(val[:-1]) * 1_000
    elif val.endswith("M"):
        return float(val[:-1]) * 1_000_000
    elif val.endswith("B"):
        return float(val[:-1]) * 1_000_000_000
    try:
        return float(val)
    except ValueError:
        return np.nan

btc["Volume"]  = btc["Vol."].apply(parse_volume)
nvda["Volume"] = nvda["Vol."].apply(parse_volume)

# 'Change %' → strip "%" and convert to float
btc["Change_pct"]  = btc["Change %"].astype(str).str.replace("%", "").astype(float)
nvda["Change_pct"] = nvda["Change %"].astype(str).str.replace("%", "").astype(float)

# Drop original raw columns
btc  = btc.drop(columns=["Vol.", "Change %"])
nvda = nvda.drop(columns=["Vol.", "Change %"])

print("\nCleaned BTC:")
print(btc.head())

print("\nCleaned NVDA:")
print(nvda.head())

# ---------------------------------------------------------------
# 3. DATE VARIABLE ENGINEERING
# ---------------------------------------------------------------

# ------ 3a. Weekday number (0=Monday … 6=Sunday) ------
btc["weekday_num"]   = btc["Date"].dt.dayofweek
nvda["weekday_num"]  = nvda["Date"].dt.dayofweek

# ------ 3b. Weekday name ------
btc["weekday_name"]  = btc["Date"].dt.day_name()
nvda["weekday_name"] = nvda["Date"].dt.day_name()

# ------ 3c. Other useful date components ------
for df, name in [(btc, "BTC"), (nvda, "NVDA")]:
    df["year"]  = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["week"]  = df["Date"].dt.isocalendar().week.astype(int)
    print(f"\n{name} – sample date features:")
    print(df[["Date", "weekday_num", "weekday_name", "year", "month", "week"]].head(7))

# ------ 3d. Remove weekends (keep only trading days Mon–Fri) ------
btc_trading  = btc[btc["weekday_num"] < 5].copy()
nvda_trading = nvda[nvda["weekday_num"] < 5].copy()

print(f"\nBTC  rows before weekend filter: {len(btc)},  after: {len(btc_trading)}")
print(f"NVDA rows before weekend filter: {len(nvda)}, after: {len(nvda_trading)}")

# ---------------------------------------------------------------
# 4. JOINING TABLES
# ---------------------------------------------------------------

# Rename price columns before joining to avoid ambiguity
btc_trading  = btc_trading.rename(columns={"Price": "BTC_Price",  "Volume": "BTC_Volume",
                                            "Change_pct": "BTC_Change",
                                            "Open": "BTC_Open", "High": "BTC_High", "Low": "BTC_Low"})
nvda_trading = nvda_trading.rename(columns={"Price": "NVDA_Price", "Volume": "NVDA_Volume",
                                             "Change_pct": "NVDA_Change",
                                             "Open": "NVDA_Open", "High": "NVDA_High", "Low": "NVDA_Low"})

# -- Keep only the columns we need for joining
btc_slim  = btc_trading[["Date", "weekday_name", "BTC_Price",  "BTC_Volume",  "BTC_Change"]]
nvda_slim = nvda_trading[["Date", "NVDA_Price", "NVDA_Volume", "NVDA_Change"]]

# ---- 4a. INNER JOIN  – only dates present in BOTH datasets ----
inner_df = pd.merge(btc_slim, nvda_slim, on="Date", how="inner")
print(f"\nInner join rows: {len(inner_df)}")
print(inner_df.head())

# ---- 4b. LEFT JOIN  – all BTC dates, NVDA filled with NaN where missing ----
left_df = pd.merge(btc_slim, nvda_slim, on="Date", how="left")
print(f"\nLeft join rows: {len(left_df)}")
print(left_df.tail())

# ---- 4c. RIGHT JOIN – all NVDA dates, BTC filled with NaN where missing ----
right_df = pd.merge(btc_slim, nvda_slim, on="Date", how="right")
print(f"\nRight join rows: {len(right_df)}")

# ---- 4d. OUTER JOIN – union of all dates from both datasets ----
outer_df = pd.merge(btc_slim, nvda_slim, on="Date", how="outer")
print(f"\nOuter join rows: {len(outer_df)}")

# Check missing values after outer join
print("\nMissing values in outer join:")
print(outer_df.isnull().sum())

# ---------------------------------------------------------------
# 5. SUMMARY STATISTICS (on inner-joined trading data)
# ---------------------------------------------------------------

df = inner_df.copy()

print("\nSummary statistics:")
print(df[["BTC_Price", "NVDA_Price", "BTC_Change", "NVDA_Change"]].describe())

mean_btc_price  = df["BTC_Price"].mean()
std_btc_price   = df["BTC_Price"].std()
mean_nvda_price = df["NVDA_Price"].mean()
std_nvda_price  = df["NVDA_Price"].std()

print(f"\nMean BTC Price : {mean_btc_price:,.2f}   Std: {std_btc_price:,.2f}")
print(f"Mean NVDA Price: {mean_nvda_price:.2f}    Std: {std_nvda_price:.2f}")

# Average change by weekday
print("\nAverage BTC daily change (%) by weekday:")
print(df.groupby("weekday_name")["BTC_Change"].mean().round(2))

# ---------------------------------------------------------------
# 6. VISUALIZATIONS
# ---------------------------------------------------------------

# --- 6a. BTC price over time ---
plt.figure()
plt.plot(df["Date"], df["BTC_Price"], color="orange")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Bitcoin Price Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 6b. NVDA price over time ---
plt.figure()
plt.plot(df["Date"], df["NVDA_Price"], color="green")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("NVIDIA Price Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 6c. Dual-axis: BTC vs NVDA ---
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(df["Date"], df["BTC_Price"],  color="orange", label="BTC")
ax2.plot(df["Date"], df["NVDA_Price"], color="green",  label="NVDA")
ax1.set_xlabel("Date")
ax1.set_ylabel("BTC Price (USD)", color="orange")
ax2.set_ylabel("NVDA Price (USD)", color="green")
plt.title("BTC vs NVDA Price (dual axis)")
plt.tight_layout()
plt.show()

# --- 6d. Histogram: daily % change ---
plt.figure()
plt.hist(df["BTC_Change"].dropna(),  bins=8, alpha=0.6, label="BTC",  color="orange")
plt.hist(df["NVDA_Change"].dropna(), bins=8, alpha=0.6, label="NVDA", color="green")
plt.xlabel("Daily Change (%)")
plt.ylabel("Frequency")
plt.title("Distribution of Daily % Change")
plt.legend()
plt.show()

# --- 6e. Scatter: BTC vs NVDA daily % change ---
plt.figure()
plt.scatter(df["BTC_Change"], df["NVDA_Change"], alpha=0.7)
plt.xlabel("BTC Daily Change (%)")
plt.ylabel("NVDA Daily Change (%)")
plt.title("BTC vs NVDA Daily % Change")
plt.show()

# ---------------------------------------------------------------
# 7. CORRELATION ANALYSIS
# ---------------------------------------------------------------

corr_vars = df[["BTC_Price", "NVDA_Price", "BTC_Change", "NVDA_Change"]].dropna()

# Pearson
pearson_corr = corr_vars.corr(method="pearson")
print("\nPearson correlation matrix:")
print(pearson_corr.round(3))

# Spearman
spearman_corr = corr_vars.corr(method="spearman")
print("\nSpearman correlation matrix:")
print(spearman_corr.round(3))

# Pairwise test: BTC change vs NVDA change
r_p, p_p = pearsonr(df["BTC_Change"].dropna(), df["NVDA_Change"].dropna())
r_s, p_s = spearmanr(df["BTC_Change"].dropna(), df["NVDA_Change"].dropna())

print(f"\nBTC Change vs NVDA Change:")
print(f"  Pearson  r={r_p:.3f}  p={p_p:.4f}")
print(f"  Spearman ρ={r_s:.3f}  p={p_s:.4f}")

# ---------------------------------------------------------------
# 8. SIMPLE OLS REGRESSION
# ---------------------------------------------------------------

df_reg = df[["NVDA_Change", "BTC_Change"]].dropna()

Y = df_reg["NVDA_Change"]       # Dependent: NVDA daily return
X = df_reg["BTC_Change"]        # Independent: BTC daily return
X = sm.add_constant(X)

model_simple = sm.OLS(Y, X).fit()
print("\n--- Simple OLS: NVDA_Change ~ BTC_Change ---")
print(model_simple.summary())

# ---------------------------------------------------------------
# 9. MULTIPLE OLS REGRESSION
# ---------------------------------------------------------------

# Add weekday dummies as additional predictors
df["Mon"] = (df["weekday_name"] == "Monday").astype(int)
df["Tue"] = (df["weekday_name"] == "Tuesday").astype(int)
df["Wed"] = (df["weekday_name"] == "Wednesday").astype(int)
df["Thu"] = (df["weekday_name"] == "Thursday").astype(int)
# Friday is the reference category (omitted to avoid multicollinearity)

df_mreg = df[["NVDA_Change", "BTC_Change", "Mon", "Tue", "Wed", "Thu"]].dropna()

Y = df_mreg["NVDA_Change"]
X = df_mreg[["BTC_Change", "Mon", "Tue", "Wed", "Thu"]]
X = sm.add_constant(X)

model_multi = sm.OLS(Y, X).fit()
print("\n--- Multiple OLS: NVDA_Change ~ BTC_Change + Weekday Dummies ---")
print(model_multi.summary())

# Visual: predicted vs actual (simple model)
df_reg2 = df[["NVDA_Change", "BTC_Change"]].dropna().copy()
X2 = sm.add_constant(df_reg2["BTC_Change"])
df_reg2["predicted"] = model_simple.predict(X2)

plt.figure()
plt.scatter(df_reg2["BTC_Change"], df_reg2["NVDA_Change"], label="Actual", alpha=0.7)
plt.plot(df_reg2["BTC_Change"].sort_values(),
         df_reg2.set_index("BTC_Change")["predicted"].sort_index(),
         color="red", label="OLS fit")
plt.xlabel("BTC Daily Change (%)")
plt.ylabel("NVDA Daily Change (%)")
plt.title("OLS Regression: NVDA on BTC")
plt.legend()
plt.show()

# ---------------------------------------------------------------
# 10. REGRESSION DIAGNOSTICS
# ---------------------------------------------------------------

residuals = model_simple.resid
fitted    = model_simple.fittedvalues

# Residuals vs fitted
plt.figure()
plt.scatter(fitted, residuals, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()

# Distribution of residuals
plt.figure()
plt.hist(residuals, bins=6)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.show()

# Breusch-Pagan test for heteroskedasticity
from statsmodels.stats.diagnostic import het_breuschpagan

X_diag = model_simple.model.exog
bp_test = het_breuschpagan(residuals, X_diag)
labels  = ["LM Statistic", "LM p-value", "F Statistic", "F p-value"]
print("\nBreusch-Pagan test:")
print(dict(zip(labels, bp_test)))

# Robust standard errors (HC3) if heteroskedasticity is detected
robust = model_simple.get_robustcov_results(cov_type="HC3")
print("\nRobust OLS summary (HC3):")
print(robust.summary())
