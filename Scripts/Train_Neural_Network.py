from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler, label_binarize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from joblib import dump

# Load the dataset from the CSV file
file_path = "Data/train_test_dataset.csv"
df = pd.read_csv(file_path)


# Separate features (X) and target (y)
X = df.drop(columns=["Class"]).values  # Drop the "Class" column to get the features
y = df["Class"].values   

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the MLP model with initial hyperparameters for testing
model = MLPClassifier(
    hidden_layer_sizes=(100,),  # Single hidden layer with 100 neurons
    activation='relu',          # Activation function
    solver='adam',              # Solver for weight optimization
    learning_rate_init=0.001,   # Initial learning rate
    alpha=0.0001,               # L2 regularization term
    max_iter=10000,             # Maximum number of iterations
    random_state=42             # Random state for reproducibility
)

# Fit the initial model
model.fit(X_train, y_train)

# Evaluate the initial model
y_pred = model.predict(X_test)
print("Initial Accuracy:", accuracy_score(y_test, y_pred))

# Define the parameter grid for GridSearchCV
param_grid = {
    'hidden_layer_sizes': [ (50, ), (100, ) ,(200,)],
    'activation': ['relu'],
    'solver': ['adam'],
    'learning_rate_init': [0.001, 0.01],
    'alpha': [0.0001]
}

# Apply GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    MLPClassifier(max_iter=1000, random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2  # Verbose output for tracking progress
)

print("Starting Grid Search...")
grid_search.fit(X_train, y_train)

# Extract the best model and parameters from the grid search
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate the best model
y_pred_best = best_model.predict(X_test)

# Create a confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_best)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best, average="weighted")
recall = recall_score(y_test, y_pred_best, average="weighted")
f1 = f1_score(y_test, y_pred_best, average="weighted")


# Print evaluation metrics
print(f"Best Parameters: {best_params}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save confusion matrix as an image
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title(f"Confusion Matrix (Params: {best_params})")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(np.arange(len(np.unique(y))), np.unique(y))
plt.yticks(np.arange(len(np.unique(y))), np.unique(y))
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j], ha="center", va="center", color="red")

# Create results directory if not exists
os.makedirs("Results", exist_ok=True)
conf_matrix_path = f"Results/Neural_Network_{best_params}.png"
plt.savefig(conf_matrix_path)
plt.close()
print(f"Confusion matrix saved to {conf_matrix_path}")

# Save classification report as JSON
class_report = classification_report(y_test, y_pred_best, output_dict=True)
json_report_path = f"Results/Neural_Network_{best_params}.json"
with open(json_report_path, "w") as json_file:
    json.dump(class_report, json_file, indent=4)
print(f"Classification report saved to {json_report_path}")

# Save the trained model
model_path = "Models/Neural_Network.pkl"
dump(best_model, model_path)

print(f"Model saved to {model_path}")


