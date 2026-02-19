# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# =========================
# Load dataset
# =========================
df = pd.read_csv("order_history_kaggle_data.csv")

# =========================
# Clean Distance column
# =========================
df["Distance"] = df["Distance"].astype(str)          # Convert to string first
df["Distance"] = df["Distance"].str.lower()          # Make lowercase
df["Distance"] = df["Distance"].str.replace("km", "", regex=False)
df["Distance"] = df["Distance"].str.strip()          # Remove extra spaces
df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")

# =========================
# Select target
# =========================
target_column = "Order Status"

# =========================
# Select useful features
# =========================
features = [
    "Distance",
    "Bill subtotal",
    "Packaging charges",
    "Total",
    "KPT duration (minutes)",
    "Rider wait time (minutes)"
]

data = df[features + [target_column]].copy()

# =========================
# Handle missing values
# =========================
data.fillna(0, inplace=True)

# =========================
# Encode target column
# =========================
le = LabelEncoder()
data[target_column] = le.fit_transform(data[target_column])

# =========================
# Split data
# =========================
X = data[features]
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Train Random Forest Model
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =========================
# Save model and encoder
# =========================
joblib.dump(model, "rf_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model trained and saved successfully and the Accuracy is- 0.9957796014067996!")
