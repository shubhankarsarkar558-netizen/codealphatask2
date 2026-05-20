import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("task2.csv")

# First 5 rows
print(df.head())

# Dataset shape
print("Shape:", df.shape)

# Dataset info
print(df.info())

# Missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

# Survival count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Passenger class count
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class")
plt.show()

# Gender distribution
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")
plt.show()

# Age distribution
sns.histplot(df['Age'].dropna(), bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()