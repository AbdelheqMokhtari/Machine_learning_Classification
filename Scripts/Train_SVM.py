import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import os

# Function to remove outliers using Isolation Forest
def remove_outliers(X, y):
    iso = IsolationForest(contamination=0.05, random_state=42)
    outlier_predictions = iso.fit_predict(X)
    mask = outlier_predictions == 1
    return X[mask], y[mask]

# Step 1: Load the main dataset from CSV
train_test_file_path = "Data/train_test_dataset.csv"
df_train_test = pd.read_csv(train_test_file_path)

# Separate features and target for the main dataset
X = df_train_test.drop("Class", axis=1)
y = df_train_test["Class"]

# Remove outliers from the main dataset
X, y = remove_outliers(X.values, y.values)

# Split the main dataset into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocess the main dataset: Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Hyperparameter tuning
print("Tuning hyperparameters on the main dataset...")
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly']
}
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")
svm_model = grid_search.best_estimator_

# Step 3: Train the SVM model on the main dataset using the best parameters
print("\nTraining on main dataset with best parameters...")
svm_model.fit(X_train, y_train)

# Evaluate on the main test set
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on main test data: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# # Step 4: Save the trained model
# models_dir = "models"
# os.makedirs(models_dir, exist_ok=True)
# model_file_path = os.path.join(models_dir, "svm_model.joblib")
# joblib.dump(svm_model, model_file_path)

# print(f"Model has been saved to {model_file_path}")
