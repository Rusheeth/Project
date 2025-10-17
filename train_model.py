# train_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
X, y = load_iris(return_X_y=True)

# Train simple model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")
