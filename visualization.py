# Data Visualization with Matplotlib & Seaborn - Vaibhavi Swami | IC-2026-3303
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
df['Age'] = df['Age'].fillna(df['Age'].median())

sns.set_style("whitegrid")

plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True, color='teal')
plt.title('Histogram - Age Distribution')
plt.tight_layout(); plt.show()

plt.figure(figsize=(7,5))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='magma')
plt.title('Bar Chart - Survival Count by Pclass')
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, palette='coolwarm', alpha=0.7)
plt.title('Scatter Plot - Age vs Fare by Survival')
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Pclass', y='Fare', data=df, palette='Set2')
plt.title('Box Plot - Fare Distribution by Pclass')
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,6))
corr = df[['Survived','Pclass','Age','SibSp','Parch','Fare']].corr()
sns.heatmap(corr, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap - Titanic Features')
plt.tight_layout(); plt.show()
