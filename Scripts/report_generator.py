import os
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


file_path = 'Data/train_test_dataset.csv'  
df = pd.read_csv(file_path)


target_column = 'Class'
X = df.drop(columns=[target_column])
y = df[target_column]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


os.makedirs('Results', exist_ok=True)


model_paths = [
    "Models/Neural_Network.pkl",  
    "Models/random_forest.pkl",    
    "Models/svm_model.joblib"     
]


def generate_results(model_path, X_test, y_test):
    
    if model_path.endswith(".joblib"):
        model = joblib.load(model_path)
    else:
        model = joblib.load(model_path)
    
    
    y_pred = model.predict(X_test)
    
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.title(f"Confusion Matrix: {model_path}")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(np.arange(len(np.unique(y_test))), np.unique(y_test))
    plt.yticks(np.arange(len(np.unique(y_test))), np.unique(y_test))
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, conf_matrix[i, j], ha="center", va="center", color="red")
    conf_matrix_image_path = f"Results/{model_path.split('/')[-1].split('.')[0]}_confusion_matrix.png"
    plt.savefig(conf_matrix_image_path)
    plt.close()
    print(f"Confusion matrix saved to {conf_matrix_image_path}")
    
    
    class_report_json_path = f"Results/{model_path.split('/')[-1].split('.')[0]}_classification_report.json"
    with open(class_report_json_path, 'w') as json_file:
        json.dump(class_report, json_file, indent=4)
    print(f"Classification report saved to {class_report_json_path}")


for model_path in model_paths:
    generate_results(model_path, X_test_scaled, y_test)
