import shap
import joblib
import pandas as pd

model = joblib.load("models/credit_model.pkl")

df = pd.read_csv("data/german_credit_data.csv")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(df.drop("Risk", axis=1))

shap.summary_plot(shap_values, df.drop("Risk", axis=1))