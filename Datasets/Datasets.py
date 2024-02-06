import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.io
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.datasets import make_circles, make_moons, make_blobs, fetch_openml
import matplotlib.pyplot as plt
from mnist import MNIST
SHUFFLE = True 
IMG_SIZE = 28

folder_path = './Datasets/'

def get_pendigits_np():
    data = np.vstack([
        np.loadtxt(folder_path + "Pendigits/pendigits.tra", delimiter=','),
        np.loadtxt(folder_path + "Pendigits/pendigits.tes", delimiter=',')
    ])
    pendigits, labels = data[:, :-1], data[:, -1:]

    # Fix pendigits
    pendigits = pendigits.astype("float")
    scaler = MinMaxScaler()
    pendigits = scaler.fit_transform(pendigits)

    # Fix labels
    labels = np.squeeze(labels)
    labels = labels.astype("int")

    return pendigits, labels

def get_har_np():
    df_train = pd.read_csv(folder_path + 'Har/train.csv', index_col=0)
    df_test = pd.read_csv(folder_path + 'Har/test.csv', index_col=0)
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    del df_train, df_test
    
    labels = df['Activity']
    df.drop(columns=['subject','Activity'], inplace=True)
    data = np.array(df)
    labels = np.squeeze(np.array(labels))

    data = MinMaxScaler().fit_transform(data)
    labels = LabelEncoder().fit_transform(labels)

    return data, labels    
    
def get_synthetic_np():
    np.random.seed(42)
 
    # Elements for high projections (DCN) 
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    W = np.random.normal(loc=0.0, scale=1.0, size=(2, 10))
    U = np.random.normal(loc=0.0, scale=1.0, size=(10, 100))
     
    # Number of samples and dimensions
    n_samples = 2500
    n_features = 2
     
    center_1 = np.full(shape=(1, n_features), fill_value=0)
    center_2 = np.full(shape=(1, n_features), fill_value=0)
    center_3 = np.full(shape=(1, n_features), fill_value=0)
    center_4 = np.full(shape=(1, n_features), fill_value=0)
     
    center_2[0][0] = 6
    center_3[0][1] = 6
     
    center_4[0][0] = 6
    center_4[0][1] = 6
     
    # Create data for each Gaussian cluster
    cluster1 = make_blobs(n_samples=n_samples, n_features=n_features, centers=center_1, cluster_std=1.0)
    cluster2 = make_blobs(n_samples=n_samples, n_features=n_features, centers=center_2, cluster_std=1.0)
    cluster3 = make_blobs(n_samples=n_samples, n_features=n_features, centers=center_3, cluster_std=1.0)
    cluster4 = make_blobs(n_samples=n_samples, n_features=n_features, centers=center_4, cluster_std=1.0)
     
    # Combine the clusters to form a 2x2 grid
    data = np.vstack([cluster1[0], cluster2[0], cluster3[0], cluster4[0]])
    data = sigmoid(sigmoid(data @ W) @ U)
    data = MinMaxScaler().fit_transform(data).astype(np.float32)
    labels = np.hstack([np.zeros(n_samples), np.ones(n_samples), 2 * np.ones(n_samples), 3 * np.ones(n_samples)])
    labels = LabelEncoder().fit_transform(labels)

    return data, labels

def get_emnist_general_np(option):
    # Options: balanced, byclass, bymerge, digits, letters, mnist
    mndata = MNIST(folder_path + 'Emnist')
    if option == 'mnist':
        mndata.select_emnist('mnist')
    else:    
        mndata.select_emnist('balanced') 
    data, labels = mndata.load_training()
    data_ts, labels_ts = mndata.load_testing()

    data = np.vstack((data, data_ts))
    labels = np.hstack((labels, labels_ts))
    data = MinMaxScaler().fit_transform(data).astype(np.float32)

    data = np.reshape(data, (-1, 1, IMG_SIZE, IMG_SIZE))
    labels = np.array(labels)
    labels = LabelEncoder().fit_transform(labels)
    
    return data, labels

def get_emnist_balanced_letters_A_J_np():
    data, labels = get_emnist_general_np('letters')
    valid_indices = np.where((labels >=10 ) & (labels < 20))[0]
    
    data = data[valid_indices] 
    labels = labels[valid_indices]
    
    return data, labels

def get_emnist_balanced_letters_K_T_np():
    data, labels = get_emnist_general_np('letters')
    valid_indices = np.where((labels >=20 ) & (labels < 30))[0]
    
    data = data[valid_indices] 
    labels = labels[valid_indices]
    
    return data, labels

def get_emnist_balanced_letters_U_Z_np():
    data, labels = get_emnist_general_np('letters')
    valid_indices = np.where((labels >=30 ) & (labels < 36))[0]
    
    data = data[valid_indices] 
    labels = labels[valid_indices]
    
    return data, labels

def get_emnist_balanced_digits_np():
    data, labels = get_emnist_general_np('digits')
    valid_indices = np.where(labels < 10)[0]

    data = data[valid_indices] 
    labels = labels[valid_indices]
    
    return data, labels

def get_emnist_mnist_np():
    return get_emnist_general_np('mnist')

def get_waveform_v1_np():
    df_data = pd.read_csv(folder_path + 'Waveform-v1/data.csv', header = None)
    df_labels = pd.read_csv(folder_path + 'Waveform-v1/labels.csv', header = None)
 
    data = np.array(df_data)
    labels = df_labels.to_numpy()
    data = MinMaxScaler().fit_transform(data)
    labels = LabelEncoder().fit_transform(labels)
 
    return data, labels

def get_dataset(dataset_name, batch_size=64):
    function_name = "get_" + dataset_name + "_np"
    function_to_call = globals()[function_name]
    
    data_np, labels_np = function_to_call()
    
    # Convert to tensor dataset
    data = torch.Tensor(data_np)
    data_shape = data.shape[1]
    labels = torch.Tensor(labels_np)
    final_dataset = TensorDataset(data, labels)
    dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

    return dataloader, data_shape, data_np, labels_np
