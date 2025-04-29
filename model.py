import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("Indian_housing_Delhi_data.csv")

# Clean house_size
df["house_size"] = (
    df["house_size"]
    .str.replace("sq ft", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
    .astype(float)
)

# Define features and target
features = ["house_size", "location", "city", "numBathrooms", "numBalconies"]
target = "price"

X = df[features].copy()
y = df[target]

# Encode categorical features
label_encoders = {}
for col in ["location", "city"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "house_price_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("✅ Model trained and saved.")
