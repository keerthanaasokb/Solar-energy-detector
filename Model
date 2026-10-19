"""Predict a day's solar output level (Low / Medium / High) from the weather.

Models: Decision Tree and K-Nearest Neighbors. Run: python solar_model.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---- 2. Data preparation -------------------------------------------------
df = pd.read_csv("solar_data.csv")
print("Rows, columns:", df.shape)
print("\nMissing values per column:\n", df.isna().sum()[df.isna().sum() > 0])
print("\nClass balance:\n", df["output_level"].value_counts())

# energy_kwh is what the label is built from, so using it as a feature would be cheating
features = ["month", "cloud_cover_pct", "temp_max_c", "temp_min_c",
            "humidity_pct", "rain_mm", "wind_kmh", "day_length_h"]
X, y = df[features], df["output_level"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---- 3. Model building ---------------------------------------------------
# Imputing inside the pipeline means medians come from training data only.
tree = Pipeline([("fill", SimpleImputer(strategy="median")),
                 ("model", DecisionTreeClassifier(random_state=42))])
knn = Pipeline([("fill", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),          # KNN depends on distances
                ("model", KNeighborsClassifier())])

tree_search = GridSearchCV(tree, {"model__max_depth": [2, 3, 4, 5, 6, 8]}, cv=5).fit(X_train, y_train)
knn_search = GridSearchCV(knn, {"model__n_neighbors": [3, 5, 7, 9, 15, 25]}, cv=5).fit(X_train, y_train)
print("\nBest tree settings:", tree_search.best_params_)
print("Best KNN settings :", knn_search.best_params_)

# ---- 4. Evaluation -------------------------------------------------------
labels = ["Low", "Medium", "High"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
results = {}
for ax, (name, search) in zip(axes, [("Decision Tree", tree_search), ("KNN", knn_search)]):
    pred = search.predict(X_test)
    results[name] = accuracy_score(y_test, pred)
    print(f"\n=== {name}: test accuracy {results[name]:.3f} ===")
    print(classification_report(y_test, pred, labels=labels))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, pred, labels=labels))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=labels, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} (accuracy {results[name]:.1%})")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)

imp = pd.Series(tree_search.best_estimator_["model"].feature_importances_, index=features).sort_values(ascending=False)
print("\nDecision Tree feature importance:\n", imp.round(3))

# ---- 5. Conclusion -------------------------------------------------------
best = max(results, key=results.get)
print(f"\nBetter model on the test set: {best}")
