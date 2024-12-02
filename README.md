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

   The trained models will be saved in the `Models/` directory.

3. **Visualize Data and Results**  
   - Use `data_visualization.ipynb` to generate and explore different datasets with various parameters.  
   - Use `Result_Visualisation.ipynb` to load and compare the performance of the trained models on the validation dataset.

4. **Make Predictions**  
   Use `predict_classification.py` to make predictions with the trained models. The script can accept arguments to display results for a specific model. For example:  
   ```bash
   python predict_classification.py --model "Neural Network"

   python predict_classification.py --model "SVM"

   python predict_classification.py --model "Random Forest"
    ```


   If no arguments are provided, the script will display results for all three models by default.

5. **Generate Reports**
Run `report_generation.py` to produce performance reports and visualizations for each model. These will be saved in the `Results/` directory.

## Model Evaluation and Hyperparameter Tuning

To ensure the best performance, we implemented **5-fold cross-validation** in the three scripts `predict_classification.py`, `RandomForest.py` and `Train_Neural_Network.py` to evaluate the models. During this process, we tested different hyperparameters using **GridSearchCV** to select the best model.

- **5-Fold Cross-Validation**: The dataset is split into five folds, and the model is trained and evaluated five times, each time using a different fold as the validation set and the remaining folds as the training set. This helps to assess the model’s performance more robustly.

- **GridSearchCV**: This technique is used to search through a specified hyperparameter grid to find the optimal parameters for each model. It performs an exhaustive search over all possible combinations of hyperparameters, ensuring the selection of the best-performing model based on cross-validation results.

By using 5-fold cross-validation with GridSearch, we ensure that the model selected is the most reliable and robust with respect to the dataset.


## Results

Each model's performance is stored as:  
- `.png` files for visualizations  
- `.json` files for numerical metrics  

These results are generated using the `report_generation.py` script and can be compared in detail using the `Result_Visualisation.ipynb` notebook.

## Docker

This project uses Docker to containerize the application and ensure a consistent environment for running the machine learning model training and evaluation tasks.

### Setup and Run with Docker

1. **Build the Docker Image**  
   To build the Docker image for this project, use the following command:

   ```bash
   docker build -t ml-models .

2. **Run the Docker Container**
    After building the image, you can run the container using:

    ```bash
    docker run -it --rm ml-models

3. **Use Docker Compose**
    For easier management, the project also includes a docker-compose.yml file. You can start the container with:

    ```bash
    docker-compose up --build

4. **Docker Volumes**
    The project uses Docker volumes to link the Models directory on your local machine with the Models directory inside the container. This ensures that any models trained or updated inside the container are reflected on your local machine.

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
