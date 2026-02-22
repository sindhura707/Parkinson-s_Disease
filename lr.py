import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('parkinsons.data')
print(data.head())

numeric_data = data.drop('name', axis=1)

sns.countplot(x='status', data=numeric_data, color='violet')
plt.title('Distribution of Status')
plt.show()

plt.figure(figsize=(15, 10))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", color='lime')
plt.title('Correlation Heatmap')
plt.show()

target_variable_column_name = 'status'
X = numeric_data.drop(target_variable_column_name, axis=1)
y = numeric_data[target_variable_column_name]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('\nClassification Report:')
print(classification_report(y_test, y_pred))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))