import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score)
from scipy.cluster.hierarchy import linkage, fcluster

np.random.seed(42)
report = []

# =================================================================
# PART A — CLUSTERING (K-Means, Hierarchical) + DIMENSIONALITY REDUCTION
# =================================================================
df = pd.read_csv("employee_cleaned.csv")

# Use numeric predictor features only (unsupervised — no target used)
cluster_features = ["Age", "YearsAtCompany", "MonthlyIncome",
                     "SatisfactionScore", "Education_Encoded"]
X = df[cluster_features].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

report.append("CLUSTERING FEATURES\n" + ", ".join(cluster_features))

# ---------- Dimensionality reduction with PCA ----------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_
report.append("PCA — explained variance ratio\nPC1: {:.3f}, PC2: {:.3f}, Total: {:.3f}".format(
    explained[0], explained[1], explained.sum()))

# ---------- K-Means: choose k via elbow + silhouette ----------
inertias, sil_scores = [], []
k_range = range(2, 7)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

elbow_table = pd.DataFrame({"k": list(k_range), "Inertia": np.round(inertias, 1),
                             "Silhouette": np.round(sil_scores, 3)})
report.append("K-MEANS — elbow & silhouette scan\n" + elbow_table.to_string(index=False))

best_k = k_range[int(np.argmax(sil_scores))]
report.append(f"K-MEANS — best k by silhouette score: {best_k}")

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_scaled)
df["KMeans_Cluster"] = kmeans_labels

kmeans_profile = df.groupby("KMeans_Cluster")[cluster_features].mean().round(2)
kmeans_profile["Count"] = df.groupby("KMeans_Cluster").size()
report.append("K-MEANS — cluster profiles (mean feature values)\n" + kmeans_profile.to_string())

kmeans_sil = silhouette_score(X_scaled, kmeans_labels)
report.append(f"K-MEANS — final silhouette score (k={best_k}): {kmeans_sil:.3f}")

# ---------- Hierarchical (Agglomerative) Clustering ----------
hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
hier_labels = hier.fit_predict(X_scaled)
df["Hierarchical_Cluster"] = hier_labels
hier_sil = silhouette_score(X_scaled, hier_labels)
report.append(f"HIERARCHICAL CLUSTERING — silhouette score (k={best_k}, ward linkage): {hier_sil:.3f}")

hier_profile = df.groupby("Hierarchical_Cluster")[cluster_features].mean().round(2)
hier_profile["Count"] = df.groupby("Hierarchical_Cluster").size()
report.append("HIERARCHICAL — cluster profiles (mean feature values)\n" + hier_profile.to_string())

agreement = (kmeans_labels == hier_labels).mean()
report.append(f"CLUSTER AGREEMENT — fraction of points with the same label in both methods: {agreement:.2f} "
              "(label numbering is arbitrary between methods, so this is only a rough indicator)")

df.to_csv("employee_clustered.csv", index=False)

# =================================================================
# PART B — MODEL EVALUATION: CROSS-VALIDATION + HYPERPARAMETER TUNING
# =================================================================
y = df["Attrition_Encoded"] if "Attrition_Encoded" in df.columns else pd.read_csv("employee_cleaned.csv")["Attrition_Encoded"]
feat_cols = ["Age", "YearsAtCompany", "MonthlyIncome", "SatisfactionScore", "Education_Encoded",
             "Dept_Engineering", "Dept_Finance", "Dept_HR", "Dept_Marketing", "Dept_Sales"]
Xc = df[feat_cols].astype(float)

Xc_train, Xc_test, y_train, y_test = train_test_split(
    Xc, y, test_size=0.25, random_state=42, stratify=y)

# ---------- 5-fold cross-validation on the default model ----------
base_model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(base_model, Xc_train, y_train, cv=5, scoring="f1")
report.append("CROSS-VALIDATION — 5-fold F1 scores (default Random Forest)\n" +
              str(np.round(cv_scores, 3)) + f"\nMean F1: {cv_scores.mean():.3f}  (std: {cv_scores.std():.3f})")

# ---------- Hyperparameter tuning with GridSearchCV ----------
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 7, None],
    "min_samples_leaf": [1, 3, 5],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,
                     cv=5, scoring="f1", n_jobs=-1)
grid.fit(Xc_train, y_train)

report.append("HYPERPARAMETER TUNING — GridSearchCV best params\n" + str(grid.best_params_))
report.append(f"HYPERPARAMETER TUNING — best cross-val F1 during search: {grid.best_score_:.3f}")

# ---------- Final evaluation on the untouched test set ----------
best_model = grid.best_estimator_
test_preds = best_model.predict(Xc_test)
test_probs = best_model.predict_proba(Xc_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, test_preds),
    "Precision": precision_score(y_test, test_preds, zero_division=0),
    "Recall": recall_score(y_test, test_preds, zero_division=0),
    "F1-Score": f1_score(y_test, test_preds, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, test_probs),
}
report.append("FINAL TEST SET EVALUATION — tuned Random Forest\n" +
              pd.Series(metrics).round(3).to_string())

cm = confusion_matrix(y_test, test_preds)
report.append("FINAL TEST SET — confusion matrix\n" +
              pd.DataFrame(cm, index=["Actual: No", "Actual: Yes"],
                           columns=["Pred: No", "Pred: Yes"]).to_string())

# Compare untuned (Week 2 style default) vs tuned model on the same split
default_model = RandomForestClassifier(random_state=42)
default_model.fit(Xc_train, y_train)
default_preds = default_model.predict(Xc_test)
default_f1 = f1_score(y_test, default_preds, zero_division=0)
report.append(f"COMPARISON — default RF F1 on test set: {default_f1:.3f}  |  "
              f"tuned RF F1 on test set: {metrics['F1-Score']:.3f}")

with open("week3_output.txt", "w") as f:
    f.write("\n\n".join(report))

print("DONE")
print("Best k:", best_k, "| KMeans silhouette:", round(kmeans_sil, 3),
      "| Hierarchical silhouette:", round(hier_sil, 3))
print("Best params:", grid.best_params_)
print("Tuned test F1:", round(metrics["F1-Score"], 3), "vs default test F1:", round(default_f1, 3))
