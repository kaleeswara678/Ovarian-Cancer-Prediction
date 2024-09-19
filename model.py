import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data_frame = pd.read_csv('C:/Users/kalee/Downloads/EDUCATION - STUDY MATERIALS/SEMESTER 6/MINOR PROJECT -4/ovarianDataset.csv')

# Prepare the data
X = data_frame.drop('TYPE', axis=1)  # Feature matrix
y = data_frame['TYPE']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier
svm_classifier = SVC(kernel='linear', probability=True, random_state=42)
svm_classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)

# Save the trained model and related objects
with open('model.pkl', 'wb') as model_file:
    pickle.dump(svm_classifier, model_file)

with open('feature_names.pkl', 'wb') as feature_file:
    pickle.dump(X.columns.tolist(), feature_file)
