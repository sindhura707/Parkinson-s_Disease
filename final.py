import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = r'parkinsons.data'
parkinsons_data = pd.read_csv(file_path)

# Prepare data
X = parkinsons_data.drop(columns=["name", "status"])  # Exclude non-relevant columns
y = parkinsons_data["status"]  # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Gradient Boosting classifier
gb_model = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3)
gb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", confusion_mat)

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=["Healthy", "Parkinson's"], yticklabels=["Healthy", "Parkinson's"])
plt.title("Confusion Matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# 2. Feature Importance Plot
feature_importance = gb_model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importance)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(feature_importance)), np.array(X.columns)[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance for Gradient Boosting Model")
plt.show()

# 3. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, gb_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Example new data (replace with actual values)
new_data = pd.DataFrame([{
    "MDVP:Fo(Hz)": 150.0,  # Replace these values with real test data
    "MDVP:Fhi(Hz)": 200.0,
    "MDVP:Flo(Hz)": 100.0,
    "MDVP:Jitter(%)": 0.01,
    "MDVP:Jitter(Abs)": 0.0001,
    "MDVP:RAP": 0.005,
    "MDVP:PPQ": 0.007,
    "Jitter:DDP": 0.015,
    "MDVP:Shimmer": 0.03,
    "MDVP:Shimmer(dB)": 0.3,
    "Shimmer:APQ3": 0.02,
    "Shimmer:APQ5": 0.025,
    "MDVP:APQ": 0.02,
    "Shimmer:DDA": 0.03,
    "NHR": 0.02,
    "HNR": 20.0,
    "RPDE": 0.4,
    "DFA": 0.8,
    "spread1": -5.0,
    "spread2": 0.3,
    "D2": 2.5,
    "PPE": 0.4
}])

# Make prediction
prediction = gb_model.predict(new_data)

# Output the result
if prediction[0] == 1:
    print("Prediction: Parkinson's disease")
else:
    print("Prediction: Healthy")
