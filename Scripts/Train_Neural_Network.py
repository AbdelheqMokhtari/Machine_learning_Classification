from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

# Define the MLP model and use GridSearchCV for hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'alpha': [0.0001, 0.001]
}

grid_search = GridSearchCV(
    MLPClassifier(max_iter=1000, random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Define the MLP model with chosen hyperparameters
model = MLPClassifier(
    hidden_layer_sizes=(100,),  # Single hidden layer with 100 neurons
    activation='relu',          # Activation function
    solver='adam',              # Solver for weight optimization
    learning_rate_init=0.001,   # Initial learning rate
    alpha=0.0001,               # L2 regularization term
    max_iter=1000,              # Maximum number of iterations
    random_state=42             # Random state for reproducibility
)


# Fit the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
# print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))

