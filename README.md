# Machine Learning Model Training and Evaluation

This project demonstrates the training, evaluation, and visualization of three different machine learning models: Support Vector Machine (SVM), Neural Network (NN), and Random Forest (RF). 

## Project Description

We generate and use a synthetic classification dataset to train and evaluate three different machine learning models. The dataset is created using the `make_classification` function from `sklearn.datasets`. Key parameters of the dataset include:

- **Number of samples**: 10,000
- **Number of features**: 20
- **Number of informative features**: 5
- **Number of classes**: 4
- **Class distribution weights**: [0.3, 0.3, 0.2, 0.2]
- **Noise in labels (flip_y)**: 10%
- **Class separability**: 1.0

The dataset is split into two parts:
- **Train/Test Dataset (90%)**:Used to train and test the models during the training phase. Saved as `Train_Test_dataset.csv`
- **Validation Dataset (10%)**:Reserved for validation and used later in predict_classification.py to load the trained models and evaluate their performance on unseen data Saved as `validation.csv`

The dataset generation is done once and saved to CSV files to ensure consistency across runs.

## Folder Structure

The project is organized as follows:


### Folder Details

1. **Models/**  
   This directory contains the trained model files (`.pkl`) for Neural Network, and Random Forest and (`.joblib`) for SVM.


2. **Data/**  
   - `Dataset.csv`: The entire generated dataset.  
   - `Train_Test_dataset.csv`: Training and testing data (90% of the dataset).  
   - `validation.csv`: Validation data (10% of the dataset).

3. **Results/**  
   Contains visualizations and evaluation metrics in both `.png` (plots) and `.json` (key metrics) formats for each model.

4. **Notebooks/**  
   - `data_visualization.ipynb`: Jupyter notebook used to generate and explore different datasets with various parameters for `make_classification` to decide on the most suitable dataset for the project. 
   - `Result_Visualisation.ipynb`: upyter notebook for loading the trained models and comparing their performance on the validation dataset.

5. **Scripts/**  
   - `Data_Generation.py`: Script to generate Split and save the dataset.  
   - `predict_classification.py`: Script for making predictions with the trained models.  
   - `RandomForest.py`: Script to train and save the Random Forest model.  
   - `Train_Neural_Network.py`: Script to train and save the Neural Network model.  
   - `Train_SVM.py`: Script to train and save the Support Vector Machine model.

## How to Run

1. **Generate the Dataset**  
   Run `Data_Generation.py` to create the dataset files (`Train_Test_dataset.csv` and `validation.csv`).

2. **Train Models**  
   Use the following scripts to train and save models:  
   - `Train_SVM.py`: Train the SVM model.  
   - `Train_Neural_Network.py`: Train the Neural Network model.  
   - `RandomForest.py`: Train the Random Forest model.

3. **Visualize Data and Results**  
   Open and run the notebooks in the `Notebooks/` directory to explore the dataset and compare model results.

4. **Make Predictions**  
   Use `predict_classification.py` to make predictions with the trained models.

## Results

Each model's performance is stored as:  
- `.png` files for visualizations  
- `.json` files for numerical metrics  

These results are generated using the `report_generation.py` script and can be compared in detail using the `Result_Visualisation.ipynb` notebook.

## Requirements

- Python 3
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `pickle`
  - `seaborn`

Install the required libraries using:

```bash
pip install -r requirements.txt
