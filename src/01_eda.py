import pandas as pd

df = pd.read_csv("../data/water_potability.csv")

print(df.shape)
print(df.head())
print(df.isnull().sum())
import matplotlib.pyplot as plt

print(df['Potability'].value_counts())

counts = df['Potability'].value_counts().sort_index()
plt.figure(figsize=(5,4))
plt.bar(['Non-Potable (0)', 'Potable (1)'], counts.values, color=['#e74c3c', '#2ecc71'])
plt.ylabel("Number of Samples")
plt.title("Water Potability Class Distribution")
plt.savefig("../outputs/class_distribution.png")
plt.show()

import seaborn as sns

plt.figure(figsize=(9,7))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("../outputs/correlation_heatmap.png")
plt.show()