import pandas as pd 
from sklearn.preprocessing import StandardScaler
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset from the CSV file
file_path = "Data/validation_dataset.csv"
df = pd.read_csv(file_path)

# Separate features (X) and target (y)
X = df.drop(columns=["Class"]).values  # Drop the "Class" column to get the features
y = df["Class"].values   

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load the trained models
neural_network_model = load("Models/Neural_Network.pkl")
random_forest_model = load("Models/random_forest.pkl")
svm_model = load("Models/svm_model.joblib")

# Predict with Neural Network model
nn_predictions = neural_network_model.predict(X_scaled)

# Predict with Random Forest model
rf_predictions = random_forest_model.predict(X_scaled)

# Predict with SVM model
svm_predictions = svm_model.predict(X_scaled)

# Function to print accuracy, classification report and confusion matrix
def evaluate_model(predictions, model_name):
    print(f"\nEvaluation for {model_name}:")

    # Accuracy
    accuracy = accuracy_score(y, predictions)
    print(f"Accuracy: {accuracy:.4f}")

    # Classification report
    print("Classification Report:")
    print(classification_report(y, predictions))

    # Confusion matrix
    cm = confusion_matrix(y, predictions)
    print("Confusion Matrix:")
    print(cm)


# Evaluate each model
evaluate_model(nn_predictions, "Neural Network")
evaluate_model(rf_predictions, "Random Forest")
evaluate_model(svm_predictions, "SVM")



