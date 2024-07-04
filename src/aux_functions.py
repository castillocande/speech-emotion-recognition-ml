import matplotlib.pyplot as plt
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler



def plot_distribution(y, title):
    counts, bins = np.histogram(y, bins=8, range=(1, 9))
    bin_centers = np.arange(1, 9)
    plt.bar(bin_centers, counts, width=0.9, align='center')
    plt.xticks(range(1, 9))
    plt.title(title)
    plt.xlabel("Clases")
    plt.ylabel("Muestras")
    plt.show()


def one_hot_encoder(y):
    enc = OneHotEncoder(sparse_output=False)  
    return enc.fit_transform(y.reshape(-1, 1)) 


def normalization(X):
    scaler = StandardScaler()
    X_shape = X.shape
    if len(X_shape) == 3:
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(X_shape[0], X_shape[1], X_shape[2])
        return X
    else:
        print("Debería ser dimenison 3")

def SMOTE_(X, y):
    x_shape = X.shape
    if len(x_shape) ==3:
        X = X.reshape((x_shape[0], -1))
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    if len(x_shape) == 3:
        X = X.reshape((-1, x_shape[1], x_shape[2]))
    return X, y