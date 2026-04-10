import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/german_credit_data.csv")

print(df.head())
print(df.info())

# Missing values
print(df.isnull().sum())

# Target distribution
sns.countplot(x='Risk', data=df)
plt.show()
