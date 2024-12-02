import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os


# Step 1: Generate a smaller synthetic dataset
X_small, y_small = make_classification(
    n_samples=3000,          # Smaller dataset size
    n_features=20,           # Number of features
    n_informative=5,         # Number of informative features
    n_classes=4,             # Number of target classes
    n_clusters_per_class=2,  # Number of clusters per class
    weights=[0.3, 0.3, 0.2, 0.2], # Class distribution
    flip_y=0.1,              # Noise
    class_sep=1.0,           # Separation between classes
    random_state=42          # For reproducibility
)

# Split the smaller dataset into train (80%) and test (20%)
X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(
    X_small, y_small, test_size=0.2, random_state=42, stratify=y_small
)

# Step 2: Train the SVM model on the smaller dataset
print("Training on smaller synthetic dataset...")
svm_model_small = SVC(random_state=42)
svm_model_small.fit(X_train_small, y_train_small)

# Evaluate on the smaller test set
y_pred_small = svm_model_small.predict(X_test_small)
accuracy_small = accuracy_score(y_test_small, y_pred_small)
print(f"Accuracy on smaller test data: {accuracy_small:.4f}")
print("Classification Report (Smaller Dataset):")
print(classification_report(y_test_small, y_pred_small))


#
## Step 3: Load the main dataset from CSV
#train_test_file_path = "Data/train_test_dataset.csv"
#df_train_test = pd.read_csv(train_test_file_path)
#
## Separate features and target for the main dataset
#X = df_train_test.drop("Class", axis=1)
#y = df_train_test["Class"]
#
## Split the main dataset into train (80%) and test (20%)
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=42, stratify=y
#)
#
## Preprocess the main dataset: Standardize features
#scaler_main = StandardScaler()
#X_train = scaler_main.fit_transform(X_train)
#X_test = scaler_main.transform(X_test)
#
## Step 4: Train the SVM model on the main dataset
#print("\nTraining on main dataset...")
#svm_model = SVC(random_state=42)
#svm_model.fit(X_train, y_train)
#
## Evaluate on the main test set
#y_pred = svm_model.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy on main test data: {accuracy:.4f}")
#print("Classification Report (Main Dataset):")
#print(classification_report(y_test, y_pred))
#
## Step 5: Save the trained model
#models_dir = "models"
#os.makedirs(models_dir, exist_ok=True)
#model_file_path = os.path.join(models_dir, "svm_model.joblib")
#joblib.dump(svm_model, model_file_path)
#
#print(f"Model has been saved to {model_file_path}")
