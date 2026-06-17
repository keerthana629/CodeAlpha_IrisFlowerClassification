import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("IRIS FLOWER CLASSIFICATION PROJECT")

# Load Dataset
df = pd.read_csv("iris.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Graph 1 - Species Distribution
plt.figure(figsize=(8,5))
sns.countplot(x='variety', data=df)
plt.title("Species Distribution")
plt.savefig("graph1.png")
plt.show(block=False)

# Graph 2 - Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(
    x='sepal.length',
    y='petal.length',
    hue='variety',
    data=df
)
plt.title("Scatter Plot")
plt.savefig("graph2.png")
plt.show(block=False)

# Graph 3 - Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(
    df[['sepal.length','sepal.width','petal.length','petal.width']].corr(),
    annot=True
)
plt.title("Correlation Heatmap")
plt.savefig("graph3.png")
plt.show()

# Machine Learning
X = df[['sepal.length','sepal.width','petal.length','petal.width']]
y = df['variety']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nProject Completed Successfully")