from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "Dataset" / "car_data_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"
PLOT_DIR = BASE_DIR / "Visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

data = pd.read_csv(DATA_FILE)
data["Vehicle_Age"] = data["Year"].max() - data["Year"]

target = "Selling_Price"
features = ["Car_Name", "Vehicle_Age", "Present_Price", "Driven_kms",
            "Fuel_Type", "Selling_type", "Transmission", "Owner"]
X = data[features]
y = data[target]

numeric_features = ["Vehicle_Age", "Present_Price", "Driven_kms", "Owner"]
categorical_features = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=400, random_state=42, min_samples_leaf=1
    )
}

metrics = []
trained_models = {}

for name, estimator in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    trained_models[name] = (pipeline, predictions)
    metrics.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions)
    })

metrics_df = pd.DataFrame(metrics).sort_values("R2", ascending=False)
metrics_df.to_csv(OUTPUT_DIR / "model_evaluation_metrics.csv", index=False)
best_name = metrics_df.iloc[0]["Model"]
best_pipeline, best_predictions = trained_models[best_name]

pd.DataFrame({
    "Actual_Selling_Price": y_test.values,
    "Predicted_Selling_Price": best_predictions
}).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

importance = permutation_importance(
    best_pipeline, X_test, y_test, n_repeats=15, random_state=42, scoring="r2"
)
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance.importances_mean
}).sort_values("Importance", ascending=False)
importance_df.to_csv(OUTPUT_DIR / "permutation_feature_importance.csv", index=False)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, best_predictions, alpha=0.75)
limits = [min(y_test.min(), best_predictions.min()),
          max(y_test.max(), best_predictions.max())]
plt.plot(limits, limits, linestyle="--")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title(f"Actual vs Predicted Selling Price ({best_name})")
plt.tight_layout()
plt.savefig(PLOT_DIR / "actual_vs_predicted.png", dpi=180)
plt.close()

print(metrics_df.to_string(index=False))
print(f"Best model based on test R2: {best_name}")
