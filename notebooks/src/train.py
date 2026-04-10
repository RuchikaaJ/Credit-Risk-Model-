import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

from preprocessing import preprocess_data

df = pd.read_csv("data/german_credit_data.csv")

df = preprocess_data(df)

X = df.drop("Risk", axis=1)
y = df["Risk"]

# Handle imbalance
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# Save model
joblib.dump(model, "models/credit_model.pkl")