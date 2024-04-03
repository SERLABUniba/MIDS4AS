# MIDS4AS: A Multiclass IDS for Automotive Security

The repository contains code refered to the work:

_Vita Santa Barletta, Danilo Caivano, Mirko De Vincentiis, Domenico Mazzola, Azzurra Ragone_

MIDS4AS: A Multiclass IDS for Automotive Security

# Code requirements

The code relies on the following **python3.10** libs.

Packages need are:
* [Matplotlib 3.6](https://matplotlib.org/)
* [Pandas 1.5.3](https://pandas.pydata.org/)
* [Numpy 1.23.3](https://www.numpy.org/)
* [Scikit-learn 1.1.2](https://scikit-learn.org/stable/)

# Data
The datasets used for experiments are accessible from [__Car-Hacking Dataset__](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset).

# How to use
The notebook file into the directory **notebook** contains the full code. If you want to perform the Elbow method or Homogeneity score, decomment the lines.

The directory **code** contains the code including the preprocessing, classification, and main. 

Into the file *configuration* you can set the PATHs and if you want to perform the Elbow method or Homogeneity score.

We use the random state to replicate the experiments.
