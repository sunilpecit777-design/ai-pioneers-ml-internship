import pandas as pd
import numpy as np

np.random.seed(42)

# ---------------------------------------------------------------
# 1. CREATE / LOAD A SAMPLE DATASET
# ---------------------------------------------------------------
# A synthetic "Employee Attrition" dataset with realistic quirks:
# missing values, mixed types, categorical variables, and outliers.
n = 200
departments = ["Sales", "Engineering", "HR", "Marketing", "Finance"]
education = ["High School", "Bachelor's", "Master's", "PhD"]

df = pd.DataFrame({
    "EmployeeID": range(1001, 1001 + n),
    "Age": np.random.randint(21, 60, n).astype(float),
    "Department": np.random.choice(departments, n),
    "Education": np.random.choice(education, n),
    "YearsAtCompany": np.random.randint(0, 25, n).astype(float),
    "MonthlyIncome": np.random.normal(55000, 15000, n).round(2),
    "SatisfactionScore": np.random.randint(1, 6, n).astype(float),
    "Attrition": np.random.choice(["Yes", "No"], n, p=[0.2, 0.8]),
})

# Inject missing values (MCAR) into several columns
for col, frac in [("Age", 0.06), ("MonthlyIncome", 0.05),
                   ("SatisfactionScore", 0.08), ("Department", 0.03)]:
    idx = np.random.choice(df.index, size=int(frac * n), replace=False)
    df.loc[idx, col] = np.nan

# Inject a few unrealistic outliers to be caught during EDA
df.loc[df.sample(3, random_state=1).index, "MonthlyIncome"] = df["MonthlyIncome"].max() * 4
df.loc[df.sample(2, random_state=2).index, "Age"] = 99

raw_path = "employee_raw.csv"
df.to_csv(raw_path, index=False)

report = []
report.append("RAW DATA — first 5 rows\n" + df.head().to_string())
report.append("RAW DATA — shape: {}".format(df.shape))
report.append("RAW DATA — dtypes\n" + df.dtypes.to_string())
report.append("RAW DATA — missing values per column\n" + df.isnull().sum().to_string())

# ---------------------------------------------------------------
# 2. HANDLE MISSING VALUES
# ---------------------------------------------------------------
df_clean = df.copy()

# Numeric columns -> median imputation (robust to outliers)
for col in ["Age", "MonthlyIncome", "SatisfactionScore"]:
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)

# Categorical column -> mode imputation
mode_val = df_clean["Department"].mode()[0]
df_clean["Department"] = df_clean["Department"].fillna(mode_val)

report.append("AFTER IMPUTATION — missing values per column\n" + df_clean.isnull().sum().to_string())

# ---------------------------------------------------------------
# 2b. TREAT OUTLIERS (cap using IQR)
# ---------------------------------------------------------------
def cap_outliers_iqr(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series.clip(lower, upper)

before_desc = df_clean[["Age", "MonthlyIncome"]].describe().to_string()
df_clean["Age"] = cap_outliers_iqr(df_clean["Age"])
df_clean["MonthlyIncome"] = cap_outliers_iqr(df_clean["MonthlyIncome"])
after_desc = df_clean[["Age", "MonthlyIncome"]].describe().to_string()

report.append("OUTLIER TREATMENT — Age & MonthlyIncome BEFORE capping\n" + before_desc)
report.append("OUTLIER TREATMENT — Age & MonthlyIncome AFTER capping\n" + after_desc)

# ---------------------------------------------------------------
# 3. FEATURE SELECTION
# ---------------------------------------------------------------
# Drop identifier column (no predictive value)
df_clean = df_clean.drop(columns=["EmployeeID"])

# Correlation check among numeric features (drop near-duplicate/irrelevant ones if any)
numeric_cols = ["Age", "YearsAtCompany", "MonthlyIncome", "SatisfactionScore"]
corr_matrix = df_clean[numeric_cols].corr().round(2)
report.append("FEATURE SELECTION — numeric correlation matrix\n" + corr_matrix.to_string())
report.append("FEATURE SELECTION — dropped 'EmployeeID' (identifier, no predictive signal)")

# ---------------------------------------------------------------
# 4. ENCODING CATEGORICAL VARIABLES
# ---------------------------------------------------------------
# Ordinal encoding for Education (has a natural order)
education_order = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
df_clean["Education_Encoded"] = df_clean["Education"].map(education_order)

# One-hot encoding for Department (nominal, no order)
df_encoded = pd.get_dummies(df_clean, columns=["Department"], prefix="Dept")

# Binary encoding for target label
df_encoded["Attrition_Encoded"] = df_encoded["Attrition"].map({"Yes": 1, "No": 0})

report.append("ENCODING — columns after encoding\n" + ", ".join(df_encoded.columns.tolist()))
report.append("ENCODING — sample rows\n" + df_encoded[["Education", "Education_Encoded",
                                                        "Attrition", "Attrition_Encoded"]].head().to_string())

# ---------------------------------------------------------------
# 5. NORMALIZATION / FEATURE SCALING
# ---------------------------------------------------------------
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def z_score_standardize(series):
    return (series - series.mean()) / series.std()

df_final = df_encoded.copy()
df_final["Age_MinMax"] = min_max_normalize(df_final["Age"])
df_final["MonthlyIncome_Zscore"] = z_score_standardize(df_final["MonthlyIncome"])

report.append("NORMALIZATION — Age (Min-Max) & MonthlyIncome (Z-score) sample\n" +
              df_final[["Age", "Age_MinMax", "MonthlyIncome", "MonthlyIncome_Zscore"]].head().to_string())

# ---------------------------------------------------------------
# 6. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------
eda_summary = df_final[["Age", "YearsAtCompany", "MonthlyIncome",
                         "SatisfactionScore"]].describe().round(2).to_string()
report.append("EDA — descriptive statistics\n" + eda_summary)

attrition_rate = df_final["Attrition_Encoded"].mean()
report.append("EDA — overall attrition rate: {:.1%}".format(attrition_rate))

dept_attrition = df_clean.groupby("Department")["Attrition"].apply(
    lambda s: (s == "Yes").mean()).round(3)
report.append("EDA — attrition rate by department\n" + dept_attrition.to_string())

satisfaction_corr = df_final["SatisfactionScore"].corr(df_final["Attrition_Encoded"])
report.append("EDA — correlation: SatisfactionScore vs Attrition = {:.3f}".format(satisfaction_corr))

# Save final cleaned dataset
clean_path = "employee_cleaned.csv"
df_final.to_csv(clean_path, index=False)

with open("pipeline_output.txt", "w") as f:
    f.write("\n\n".join(report))

print("DONE")
print("Raw shape:", df.shape, "-> Final shape:", df_final.shape)
