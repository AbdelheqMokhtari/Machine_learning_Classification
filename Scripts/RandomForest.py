import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

# Load the CSV file
file_path = 'Data/train_test_dataset.csv'  # Update with your actual file path
df = pd.read_csv(file_path)

# Assuming the target column is named 'Class'
# Adjust if your target column has a different name
target_column = 'Class'

# Splitting features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Splitting the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],       
    'max_depth': [None, 10, 20],             
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Retrieve the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the final model with the best parameters
best_rf_model = grid_search.best_estimator_

# Make predictions
y_pred = best_rf_model.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Ensure the 'Models' directory exists
model_dir = 'Models'
os.makedirs(model_dir, exist_ok=True)

# Save the optimized model
model_path = os.path.join(model_dir, 'optimized_random_forest.pkl')
joblib.dump(best_rf_model, model_path)

print(f"Optimized model saved to {model_path}")
