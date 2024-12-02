#!/bin/bash
python Scripts/Data_Generation.py
python Scripts/RandomForest.py
python Scripts/Train_SVM.py
python Scripts/Train_Neural_Network.py
python Scripts/predict_classification.py