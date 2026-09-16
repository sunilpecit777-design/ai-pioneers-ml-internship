import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score)

np.random.seed(42)

# ---------------------------------------------------------------
# 1. LOAD THE WEEK-1 CLEANED DATASET
# ---------------------------------------------------------------
df = pd.read_csv("employee_cleaned.csv")

# Target: predict Attrition (1 = left, 0 = stayed)
y = df["Attrition_Encoded"]

# Features: drop raw / redundant / leakage-prone columns
drop_cols = ["Attrition", "Attrition_Encoded", "Education", "Age_MinMax",
             "MonthlyIncome_Zscore"]
X = df.drop(columns=drop_cols)
X = X.astype(float)

report = []
report.append("FEATURES USED FOR MODELING\n" + ", ".join(X.columns.tolist()))
report.append("TARGET DISTRIBUTION\n" + y.value_counts().to_string() +
              "\n(0 = Stayed, 1 = Left)")

# ---------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

report.append("TRAIN/TEST SPLIT\nTrain size: {}, Test size: {} (75/25 stratified split)".format(
    len(X_train), len(X_test)))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 3. TRAIN MODELS
# ---------------------------------------------------------------
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
    "K-Nearest Neighbors (k=5)": (KNeighborsClassifier(n_neighbors=5), True),
    "Decision Tree": (DecisionTreeClassifier(max_depth=4, random_state=42), False),
    "Random Forest (100 trees)": (RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), False),
}

results = []
confusion_matrices = {}
feature_importance_rf = None

for name, (model, needs_scaling) in models.items():
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled if needs_scaling else X_test

    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    probs = model.predict_proba(Xte)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": auc,
    })
    confusion_matrices[name] = cm

    if name.startswith("Random Forest"):
        feature_importance_rf = pd.Series(model.feature_importances_, index=X.columns
                                           ).sort_values(ascending=False)

results_df = pd.DataFrame(results).set_index("Model").round(3)
report.append("MODEL COMPARISON — evaluation metrics\n" + results_df.to_string())

for name, cm in confusion_matrices.items():
    report.append(f"CONFUSION MATRIX — {name}\n" +
                   pd.DataFrame(cm, index=["Actual: No", "Actual: Yes"],
                                columns=["Pred: No", "Pred: Yes"]).to_string())

report.append("RANDOM FOREST — feature importance (top 5)\n" +
              feature_importance_rf.head(5).round(3).to_string())

best_model = results_df["F1-Score"].idxmax()
report.append(f"BEST MODEL (by F1-Score on the held-out test set): {best_model}")

with open("week2_model_output.txt", "w") as f:
    f.write("\n\n".join(report))

results_df.to_csv("model_comparison_results.csv")

print("DONE")
print(results_df)
