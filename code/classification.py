from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def kmeans_learner(X_train, k_parameter, seed):
    """
    Execute the K-Means Algorithm
    Parameters
    ----------
    X_train : X_train dataset
    k_parameter : The number of the clusters

    Returns
    -------
    kmeansobj : The prediction of the K-Means

    """

    kmeansobj = KMeans(n_clusters=k_parameter, init='random', max_iter=300, n_init=10, random_state=seed)
    return kmeansobj.fit(X_train)


def kmeans_evaluate(X_test, kmeansobj):
    """
    Get a prediction using the K-Means fit object
    Parameters
    ----------
    X_test : The X_test dataset
    kmeansobj : The fit object of the K-Means

    Returns
    -------
    The prediction

    """
    return kmeansobj.predict(X_test)


def class_to_cluster(y_train, kmeans_labels, class_names):
    """
    The algorithm is based on the principle of purity of the single cluster generated. For each of them,
    the algorithm retrieves the examples belonging to that cluster, then it retrieves the class associated with each
    of these examples and finally, the majority class is assigned to the cluster.

    Parameters
    ----------
    y_train : The label of the training set
    kmeans_labels : the label predicted by the K-Means
    class_names : The class used (DoS, Fuzzy, ...)

    Returns
    -------
    clusters : The clusters of the kmeans
    classToCluster : The cluster assign at each class
    pur : The purity
    """
    clusters = set(kmeans_labels)
    classes = set(y_train)
    class_to_cluster = []
    N = 0
    pur = 0

    for c in clusters:
        clust_examples = []
        indices = np.where(kmeans_labels == c)[0]
        print(len(indices))
        for i in indices:
            clust_examples.append(y_train[i])
        maxClass = -1
        max_n_clust_example = -1

        print("\nCluster: ", c)
        for cl in classes:
            n_clust_example = clust_examples.count(cl)
            N += n_clust_example
            print(class_names[cl], ":", n_clust_example, "examples")
            if n_clust_example > max_n_clust_example:
                maxClass = cl
                max_n_clust_example = n_clust_example
        pur += max_n_clust_example
        class_to_cluster.append(maxClass)
    pur = pur / N

    return clusters, class_to_cluster, pur


def kmeans_evaluate_elbow(X_scaled, minK, maxK, seed):
    inertias = []
    for i in range(minK, maxK):
        kmeans = KMeans(n_clusters=i, init='random', max_iter=300, n_init=10, random_state=seed)
        kmeans.fit(X_scaled)
        print(i)
        inertias.append(kmeans.inertia_)

    print("Elbow method:")
    plt.plot(range(minK, maxK), inertias)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.xticks(np.arange(minK, maxK + 1, 1.0))
    plt.show()


def kmeans_evaluate_homo(X_scaled, XTest_scaled, YTest, minK, maxK, seed):
    scores = []
    best_homo = 0
    best_k = 0
    for i in range(minK, maxK):
        kmeans = KMeans(n_clusters=i, init='random', max_iter=300, n_init=10, random_state=seed)
        kmeans.fit(X_scaled)
        predictedClusters = kmeans.predict(XTest_scaled)
        score = homogeneity_completeness_v_measure(YTest, predictedClusters)
        scores.append(score)
        print(i)
        if score[0] > best_homo:
            best_homo = score[0]
            best_k = i
        print(f"Best_homo: {best_homo}")
        print(f"Best k: {best_k}")

    print("homogeneity eval:")
    plt.title('Homogeneity evaluation:')
    plt.plot(range(minK, maxK), [s[0] for s in scores], 'r', label='Homogeneity')
    plt.xlabel('Value of K')
    plt.ylabel('Homogeneity')
    plt.xticks(np.arange(minK, maxK + 1, 1.0))
    #plt.legend(loc=4)
    plt.show()

    return best_k

def split_train_test(X, y):
    """
    Perform the split with the train_test_split into 75/25%
    Parameters
    ----------
    X : The dataset to be divided
    y : The label

    Returns
    -------
    Dataset divided

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test


def scale_dataset(dataset, object_scaler=None):
    """
    Scale the data using the MinMaxScaler in the range [0, 1].
    If the object_scaler is not defined, the method perform the MinMaxScaler on the dataset specified
    and return the object. If the object is specified, perform the transform with this object on the
    dataset specified.
    Parameters
    ----------
    dataset : the dataset to be processed
    object_scaler : If None perform the MinMaxScaler else perform the transform

    Returns
    -------
    If object_scaler is None, returns the dataset processed with fit_transform and the object
    Else return the dataset scaled

    """
    from sklearn.preprocessing import MinMaxScaler

    if object_scaler is None:
        min_max_scaler = MinMaxScaler()
        data = min_max_scaler.fit_transform(dataset)
        return data, min_max_scaler
    else:
        data = object_scaler.transform(dataset)
        return data


def evaluation_results(y_test, y_pred, class_names):
    """
    Evaluate the result using the Confusion Matrix and the Classification Report
    Parameters
    ----------
    y_test : The label of the test dataset
    y_pred : The label predicted
    class_names : The class used

    Returns
    -------

    """
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion matrix:")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.show()
    print(classification_report(y_test, y_pred, target_names=class_names, output_dict=True))
