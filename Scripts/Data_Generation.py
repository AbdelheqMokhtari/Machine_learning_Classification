from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=10000,          
    n_features=20,           
    n_informative=5,                               
    n_classes=4,             
    n_clusters_per_class=2,  
    weights=[0.5, 0.3, 0.2], 
    flip_y=0.1,              
    class_sep=1.0,           
    random_state=42         
)



df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
df["Class"] = y


file_path = "Data/dataset.csv"
df.to_csv(file_path,index=False)



# Split the data into train_test (90%) and validation (10%)
X_train_test, X_val, y_train_test, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Combine X_train_test and y_train_test into a single DataFrame
df_train_test = pd.DataFrame(X_train_test, columns=[f"Feature_{i+1}" for i in range(X_train_test.shape[1])])
df_train_test["Class"] = y_train_test

# Combine X_val and y_val into a single DataFrame
df_val = pd.DataFrame(X_val, columns=[f"Feature_{i+1}" for i in range(X_val.shape[1])])
df_val["Class"] = y_val

# Save the datasets to CSV files
train_test_file_path = "Data/train_test_dataset.csv"
validation_file_path = "Data/validation_dataset.csv"

df_train_test.to_csv(train_test_file_path, index=False)
df_val.to_csv(validation_file_path, index=False)

print("Data has been split into train_test (90%) and validation (10%) datasets.")