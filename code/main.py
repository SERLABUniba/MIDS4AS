import pandas as pd
from classification import *
from preprocessing import init_preprocessing
from joblib import dump
from configuration import param
import matplotlib.pyplot as plt
from sys import exit
import numpy as np

pd.set_option('display.max_columns', 80)


def print_cluster(cluster_class, purity):
    print("\nResults:\n")
    print("Cluster: Class =", cluster_class)
    print("\n# of cluster assigned to each class:")
    print("Class 0: ", sum(1 for v in cluster_class.values() if v == 0),
          np.where(np.array(list(cluster_class.values())) == 0)[0])
    print("Class 1: ", sum(1 for v in cluster_class.values() if v == 1),
          np.where(np.array(list(cluster_class.values())) == 1)[0])
    print("Class 2: ", sum(1 for v in cluster_class.values() if v == 2),
          np.where(np.array(list(cluster_class.values())) == 2)[0])
    print("Class 3: ", sum(1 for v in cluster_class.values() if v == 3),
          np.where(np.array(list(cluster_class.values())) == 3)[0])
    print("Class 4: ", sum(1 for v in cluster_class.values() if v == 4),
          np.where(np.array(list(cluster_class.values())) == 4)[0])

    print('\nPurity= ', purity)


def main_k_means():
    class_names = ["Normal", "Dos", "RPM Spoofing", "Fuzzy", "Gear Spoofing"]

    if param.get('preprocessing'):
        dataset = init_preprocessing()
    else:
        dataset = pd.read_csv(param.get('dataset_processing'))

    X = dataset.drop(columns=['Flag'])
    y = dataset['Flag']

    print("[+] Split the dataset into train and test")
    
    # Stratify the train and test into 75-25%
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print("[+] Scale the data")
    # Apply MinMaxScaler into the range [0, 1]
    X_train, scaler = scale_dataset(X_train)

    X_test = scale_dataset(X_test, scaler)
    seed = 43
    np.random.seed(seed)

    if param.get('evaluate_homo'):
        kmeans_evaluate_elbow(X_train, 10, 30, seed)
        best_k = kmeans_evaluate_homo(X_train, X_test, y_test, 15, 30, seed)
        
        print(best_k)

    if param.get('evaluate_elbow'):
        kmeans_evaluate_elbow(X_train, 10, 30, seed)
        

    print("[+] Exec the K-Means algorithm")
    kmeans_object = kmeans_learner(X_train, param.get('k_parameter'), seed)


    cluster, class_cluster, purity = class_to_cluster(y_train, kmeans_object.labels_, class_names)
    
    cluster_class = dict(zip(cluster, class_cluster))
    print_cluster(cluster_class, purity)
    
    prediction = kmeans_evaluate(X_test, kmeans_object)
    
    y_prediction = [cluster_class.get(key) for key in prediction]
    
    evaluation_results(y_test, y_prediction, class_names)


if __name__ == '__main__':
    main_k_means()
