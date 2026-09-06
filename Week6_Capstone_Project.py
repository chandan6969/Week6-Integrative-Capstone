# 1. Import required libraries
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# 2. Download/load the public dataset
DATA_URL = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

try:
    df = pd.read_csv(DATA_URL)
except Exception as e:
    print("Online download failed.")
    print("Download the Telco Customer Churn CSV from a public source, place it beside this notebook,")
    print("and change DATA_URL to the local filename.")
    raise e

print("Dataset loaded successfully.")
print("Shape:", df.shape)
df.head()


# 3. Basic inspection
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nData types and non-null counts:")
df.info()

print("\nDescriptive statistics:")
display(df.describe(include="all").T)


# 4. Missing values and duplicates
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

missing_table = pd.DataFrame({
    "Missing Values": missing,
    "Missing Percentage": missing_pct.round(2)
})

display(missing_table)

print("Duplicate rows:", df.duplicated().sum())


# 5. Clean data types and remove identifier
df = df.copy()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("Missing TotalCharges after numeric conversion:", df["TotalCharges"].isnull().sum())

df = df.drop(columns=["customerID"])

# Remove exact duplicates if any
before = len(df)
df = df.drop_duplicates()
print("Duplicate rows removed:", before - len(df))

print("Cleaned shape:", df.shape)
display(df.head())


# 6. Target distribution
print("Churn value counts:")
print(df["Churn"].value_counts())

print("\nChurn percentages:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))


# 7. Churn distribution
plt.figure(figsize=(7, 5))
df["Churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 8. Churn by contract type
contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
display(contract_churn.round(2))

contract_churn.plot(kind="bar", figsize=(8, 5))
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# 9. Churn by tenure
plt.figure(figsize=(8, 5))
df.boxplot(column="tenure", by="Churn")
plt.title("Tenure Distribution by Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Tenure (months)")
plt.tight_layout()
plt.show()


# 10. Churn by monthly charges
plt.figure(figsize=(8, 5))
df.boxplot(column="MonthlyCharges", by="Churn")
plt.title("Monthly Charges by Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()


# 11. Numerical correlation matrix
numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
corr = df[numeric_cols].corr()

plt.figure(figsize=(7, 5))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()

display(corr.round(2))


# 12. Prepare X and y
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)


# 13. Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("Preprocessing pipeline created successfully.")


# 14. Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))
print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(3))


# 15. Logistic Regression model
logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000, random_state=42))
])

logistic_model.fit(X_train, y_train)

y_pred_lr = logistic_model.predict(X_test)
y_prob_lr = logistic_model.predict_proba(X_test)[:, 1]

print("Logistic Regression classification report:")
print(classification_report(y_test, y_pred_lr, target_names=["No Churn", "Churn"]))


# 16. Evaluate Logistic Regression
lr_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_lr),
    "Precision": precision_score(y_test, y_pred_lr),
    "Recall": recall_score(y_test, y_pred_lr),
    "F1-score": f1_score(y_test, y_pred_lr),
    "ROC-AUC": roc_auc_score(y_test, y_prob_lr)
}

lr_results = pd.DataFrame([lr_metrics]).round(4)
display(lr_results)


# 17. Logistic Regression confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=["No Churn", "Churn"]
)
disp.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.show()


# 18. Random Forest model
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest classification report:")
print(classification_report(y_test, y_pred_rf, target_names=["No Churn", "Churn"]))


# 19. Evaluate Random Forest
rf_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "Precision": precision_score(y_test, y_pred_rf),
    "Recall": recall_score(y_test, y_pred_rf),
    "F1-score": f1_score(y_test, y_pred_rf),
    "ROC-AUC": roc_auc_score(y_test, y_prob_rf)
}

rf_results = pd.DataFrame([rf_metrics]).round(4)
display(rf_results)


# 20. Compare supervised models
comparison = pd.DataFrame([
    {"Model": "Logistic Regression", **lr_metrics},
    {"Model": "Random Forest", **rf_metrics}
]).set_index("Model").round(4)

display(comparison)


# 21. ROC curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={lr_metrics['ROC-AUC']:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_metrics['ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.show()


# 22. Cross-validation for both models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_cv = cross_val_score(logistic_model, X, y, cv=cv, scoring="accuracy")
rf_cv = cross_val_score(rf_model, X, y, cv=cv, scoring="accuracy")

cv_results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Mean CV Accuracy": [lr_cv.mean(), rf_cv.mean()],
    "CV Std": [lr_cv.std(), rf_cv.std()]
}).round(4)

display(cv_results)


# 23. Prepare features for clustering
cluster_features = ["tenure", "MonthlyCharges", "TotalCharges"]

cluster_data = df[cluster_features].copy()
cluster_data = cluster_data.fillna(cluster_data.median())

scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_data)

print("Clustering features:", cluster_features)
print("Clustering matrix shape:", cluster_scaled.shape)


# 24. Select number of clusters using silhouette score
k_values = range(2, 7)
silhouette_scores = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(cluster_scaled)
    score = silhouette_score(cluster_scaled, labels)
    silhouette_scores.append(score)

cluster_evaluation = pd.DataFrame({
    "Number of Clusters": list(k_values),
    "Silhouette Score": np.round(silhouette_scores, 4)
})

display(cluster_evaluation)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("K-Means Cluster Selection")
plt.xticks(list(k_values))
plt.tight_layout()
plt.show()


# 25. Fit final K-Means model
best_k = int(k_values[np.argmax(silhouette_scores)])
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)

df["Cluster"] = kmeans.fit_predict(cluster_scaled)

final_silhouette = silhouette_score(cluster_scaled, df["Cluster"])

print("Selected number of clusters:", best_k)
print("Final silhouette score:", round(final_silhouette, 4))

display(df["Cluster"].value_counts().sort_index())


# 26. Describe customer segments
cluster_profile = df.groupby("Cluster")[cluster_features + ["MonthlyCharges"]].mean().round(2)
cluster_profile["Customers"] = df["Cluster"].value_counts().sort_index()
display(cluster_profile)


# 27. Visualize clusters using PCA
pca = PCA(n_components=2, random_state=42)
cluster_2d = pca.fit_transform(cluster_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(
    cluster_2d[:, 0],
    cluster_2d[:, 1],
    c=df["Cluster"],
    alpha=0.6
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segments Visualized with PCA")
plt.tight_layout()
plt.show()


# 28. Automated result summary
best_model = comparison["ROC-AUC"].idxmax()
best_auc = comparison.loc[best_model, "ROC-AUC"]
best_f1_model = comparison["F1-score"].idxmax()
best_f1 = comparison.loc[best_f1_model, "F1-score"]

print("Best model by ROC-AUC:", best_model, "->", best_auc)
print("Best model by F1-score:", best_f1_model, "->", best_f1)
print("Best clustering k:", best_k)
print("Silhouette score:", round(final_silhouette, 4))

print("\nChurn rate:")
print(round(y.mean() * 100, 2), "%")


