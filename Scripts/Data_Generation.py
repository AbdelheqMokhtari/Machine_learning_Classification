from sklearn.datasets import make_classification
import pandas as pd


X, y = make_classification(
    n_samples=10000,          
    n_features=20,           
    n_informative=5,                               
    n_classes=3,             
    n_clusters_per_class=2,  
    weights=[0.5, 0.3, 0.2], 
    flip_y=0.1,              
    class_sep=0.65,           
    random_state=42         
)



df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
df["Class"] = y


file_path = "Data/dataset.csv"
df.to_csv(file_path,index=False)