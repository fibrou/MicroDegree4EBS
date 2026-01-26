
## Simple variable

count = 1 + 1
print("count =", count)

## Multiple variables

count2 = 2 *8
Final_count = count2 / count
print("Final_count =", Final_count)

## Data Types & validiation

import pandas as pd

## Check current working directory
import os
print("Current working directory:", os.getcwd())

# Load the practice Excel file
df = pd.read_excel(
    r"C:\Users\ssaar\Desktop\Lecture 1\data.xlsx")
print("Data loaded successfully")
print(df.head())
# Display data types of each column
print("Data types of each column:")
print(df.dtypes)

# Making sure 'Age' and 'Income' are numeric
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Income"] = pd.to_numeric(df["Income"], errors="coerce")

# Convert 'Active' column to binary (1 for 'Yes', 0 for 'No')
df["Active"] = df["Active"].map({"Yes": 1, "No": 0})

# Convert 'JoinDate' to datetime format
df["JoinDate"] = pd.to_datetime(df["JoinDate"], errors="coerce")

# Display data types after conversion
print(df.dtypes)

# Display the cleaned data
print("Cleaned Data:", df)

# Summary statistics for numeric columns
print(df[["Age", "Income", "Active"]].describe())

# Individual statistics
mean_age = df["Age"].mean()
std_age = df["Age"].std()

mean_income = df["Income"].mean()
std_income = df["Income"].std()

active_share = df["Active"].mean()

print("Mean age:", mean_age)
print("Std age:", std_age)
print("Mean income:", mean_income)
print("Std income:", std_income)
print("Share active:", active_share)

#Simple grouped statistics (categorical × numeric)

grouped_df = df.groupby("Department")["Income"].mean()
print("Average Income by Department:")
print(grouped_df)


#Visualizations

import pandas as pd
import matplotlib.pyplot as plt

df  = pd.read_excel(
    r"C:\Users\ssaar\Desktop\Lecture 1\data.xlsx")

# Convert types (if necessary)
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
df["Active"] = df["Active"].map({"Yes": 1, "No": 0})


# Histogram of Age
plt.figure()
plt.hist(df["Age"].dropna(), bins=5)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Distribution of Age")
plt.show()

# Histogram of Income
plt.figure()
plt.hist(df["Income"].dropna(), bins=5)
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.title("Distribution of Income")
plt.show()

# Boxplot of Income by Active status
active_income = df[df["Active"] == 1]["Income"].dropna()
inactive_income = df[df["Active"] == 0]["Income"].dropna()

plt.figure()
plt.boxplot([active_income, inactive_income], labels=["Active", "Not Active"])
plt.ylabel("Income")
plt.title("Income by Activity Status")
plt.show()

# Correlation
import pandas as pd

df  = pd.read_excel(
    r"C:\Users\ssaar\Desktop\Lecture 1\data.xlsx")


## Select variables for correlation
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
df["Active"] = df["Active"].map({"Yes": 1, "No": 0})

corr_vars = df[["Age", "Income", "Active"]]

#Pearson correlation
pearson_corr = corr_vars.corr(method="pearson")
print(pearson_corr)

#Spearman correlation
spearman_corr = corr_vars.corr(method="spearman")

#Pairwise correlation with significance tests

from scipy.stats import pearsonr, spearmanr

# Age vs Income
r_pearson, p_pearson = pearsonr(df["Age"].dropna(), df["Income"].dropna())
r_spearman, p_spearman = spearmanr(df["Age"].dropna(), df["Income"].dropna())

print("Pearson r:", r_pearson, "p-value:", p_pearson)
print("Spearman rho:", r_spearman, "p-value:", p_spearman)

corr_vars.dropna().corr().round(2)

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(df["Age"], df["Income"])
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Age vs Income")
plt.show()

#Simple OLS Regression

import pandas as pd
import statsmodels.api as sm

# Load data
df  = pd.read_excel(
    r"C:\Users\ssaar\Desktop\Lecture 1\data.xlsx")

# Type conversions
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
df["Active"] = df["Active"].map({"Yes": 1, "No": 0})

# Keep only complete cases
df_reg = df[["Income", "Age"]].dropna()

## Define dependent and independent variables
Y = df_reg["Income"]      # dependent variable
X = df_reg["Age"]         # independent variable

## Add constant (intercept)
X = sm.add_constant(X)

## Run OLS model
model = sm.OLS(Y, X)
results = model.fit()

print(results.summary())

## Multiple OLS Regression
df_reg = df[["Income", "Age", "Active"]].dropna()

Y = df_reg["Income"]
X = df_reg[["Age", "Active"]]
X = sm.add_constant(X)

results = sm.OLS(Y, X).fit()
print(results.summary())

##Visual check: regression line

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(df_reg["Age"], df_reg["Income"])
plt.plot(
    df_reg["Age"],
    results.predict(X),
)
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("OLS Regression: Income on Age")
plt.show()

# Regression diagnostics (residuals, heteroskedasticity)

import pandas as pd
import statsmodels.api as sm

df  = pd.read_excel(
    r"C:\Users\ssaar\Desktop\Lecture 1\data.xlsx")

df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
df["Active"] = df["Active"].map({"Yes": 1, "No": 0})

df_reg = df[["Income", "Age", "Active"]].dropna()

Y = df_reg["Income"]
X = df_reg[["Age", "Active"]]
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
print(model.summary())

## Extract residuals and fitted values

residuals = model.resid
fitted = model.fittedvalues

##Residuals vs fitted values (linearity & variance)

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(fitted, residuals)
plt.axhline(0)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()

## Distribution of residuals (normality check)

plt.figure()
plt.hist(residuals, bins=5)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.show()

## Formal test for heteroskedasticity (Breusch–Pagan)

from statsmodels.stats.diagnostic import het_breuschpagan

bp_test = het_breuschpagan(residuals, X)

labels = ["LM Statistic", "LM p-value", "F Statistic", "F p-value"]
results_bp = dict(zip(labels, bp_test))

print(results_bp)

## What if heteroskedasticity is present?

robust_results = model.get_robustcov_results(cov_type="HC3")
print(robust_results.summary())
