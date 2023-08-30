import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader, TensorDataset, BatchSampler, RandomSampler
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.io
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.datasets import make_circles, make_moons, fetch_olivetti_faces, make_blobs
from mnist import MNIST
import matplotlib.pyplot as plt
import os
#from k import mnist
import pdb
from PIL import Image

SHUFFLE = True 

folder_path = './Datasets/' #'C:/Users/PAPASOFT INC/Desktop/SOFT_SIL/datasets/'

def get_3D_Spheres_np(batch_size=64, option_name = ''):
    num_samples = 500
    centers = [[0, 0, 0], [5, 0, 0]]
    cluster_std = [1.0, 1.0]
    data, labels = make_blobs(n_samples=num_samples, centers=centers, cluster_std=cluster_std, random_state=0)
    # Plot blobs in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    ax.scatter(x, y, z, c=labels, cmap='viridis')
    
    # Set stable viewpoint (adjust as needed)
    ax.view_init(azim=30, elev=20)

    # Adjust perspective (optional - 'ortho' reduces distortion)
    ax.set_proj_type('ortho')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    return data, labels

def get_3D_Spheres(batch_size=64, option_name = ''):
    data, labels = get_3D_Spheres_np()

    # Convert to tensor dataset
    data = torch.Tensor(data)
    data_shape = data.shape[1]
    labels = torch.Tensor(labels)
    final_dataset = TensorDataset(data, labels)
    dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)
    
    return dataloader, data_shape, data, labels

def get_moons_np():
    data, labels = make_moons(n_samples=1_000, noise=0.05, random_state=0)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = normalize_data(data)
    data = data.astype("float")
    labels = labels.astype("int")
    return data, labels

def get_Moons(batch_size=64, option_name = ''):
	data, labels = get_moons_np()

	# Convert to tensor dataset
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

	return dataloader, data_shape, data, labels
	
def get_squeezed_gauss_np():
	# Do not normilize! #2
	data, labels = make_blobs(n_samples=1000,random_state=0, centers=[(0.45, 0.5), (0.55, 0.5)], cluster_std=((0.005, 0.15), (0.005, 0.15)))
	data = data.astype("float")
	labels = labels.astype("int")
	return data, labels

def get_Squeezed_Gauss(batch_size=64, option_name = ''):
	data, labels = get_squeezed_gauss_np()

	# Convert to tensor dataset
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

	return dataloader, data_shape, data, labels
	
def get_gauss_densities_np():
	big_blob, big_blob_label = make_blobs(n_samples=500,random_state=0, centers=[(0, 0)], cluster_std=[(0.25, 1.5)])
	small_x = 3.5
	small_y = 2.5
	small_blobs, small_blobs_labels = make_blobs(n_samples=100,random_state=0, centers=[(small_x, small_y), (small_x, -small_y)], cluster_std=((0.15, 0.15), (0.15, 0.15)))
	small_blobs_labels += 1

	data = np.vstack((big_blob, small_blobs))
	labels = np.concatenate((big_blob_label, small_blobs_labels))
	scaler = MinMaxScaler()
	data = scaler.fit_transform(data)
	data = data.astype("float")
	labels = labels.astype("int")

	total_size = data.shape[0]
	random_permutation = np.random.permutation(np.arange(total_size))
	data = data[random_permutation]
	labels = labels[random_permutation]

	return data, labels

def get_Gauss_Densities(batch_size=64, option_name = ''):
	data, labels = get_gauss_densities_np()

	# Convert to tensor dataset
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

	return dataloader, data_shape, data, labels	

def get_australian_np():
	column_names = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "label"]

	df = pd.read_csv(folder_path + "Australian/australian.dat", delimiter=" ", header=None, names=column_names)
	data = df[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14"]]
	labels = df["label"]

	data = np.array(data)
	labels = np.array(labels)

	scaler = MinMaxScaler()
	data = scaler.fit_transform(data)
	data = data.astype("float")
	label_encoder = LabelEncoder()
	labels = label_encoder.fit_transform(labels)
	labels = labels.astype("int")

	total_size = data.shape[0]
	random_permutation = np.random.permutation(np.arange(total_size))
	data = data[random_permutation]
	labels = labels[random_permutation]

	return data, labels

def get_Australian(batch_size=64, option_name = ''):
	data, labels = get_australian_np()

	# Convert to tensor dataset
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

	return dataloader, data_shape, data, labels

def get_wine_np():
	column_names = ["label", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13"]
	df = pd.read_csv(folder_path + "Wine/wine.data", header=None, names=column_names)
	data = df[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13"]]
	labels = df["label"]
	data = np.array(data)
	labels = np.array(labels)

	scaler = MinMaxScaler()
	data = scaler.fit_transform(data)
	data = data.astype("float")
	label_encoder = LabelEncoder()
	labels = label_encoder.fit_transform(labels)
	labels = labels.astype("int")

	total_size = data.shape[0]
	random_permutation = np.random.permutation(np.arange(total_size))
	data = data[random_permutation]
	labels = labels[random_permutation]

	return data, labels

def get_Wine(batch_size=64, option_name = ''):
	data, labels = get_wine_np()

	# Convert to tensor dataset
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

	return dataloader, data_shape, data, labels

def get_ring_np():
	data, labels = make_circles(n_samples=1_000, factor=0.3, noise=0.05, random_state=0)
	scaler = MinMaxScaler()
	data = scaler.fit_transform(data)
	data = data.astype("float")
	labels = labels.astype("int")
	return data, labels

def get_Rings(batch_size=64, option_name = ''):
    data, labels = get_ring_np()

    # Convert to tensor dataset
    data = torch.Tensor(data)
    data_shape = data.shape[1]
    labels = torch.Tensor(labels)
    final_dataset = TensorDataset(data, labels)
    dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)
    
    return dataloader, data_shape, data, labels 
def get_iris_np():
	columnn_names = ["f1", "f2", "f3", "f4", "label"]
	df = pd.read_csv(folder_path + "Iris/iris.data", header=None, names=columnn_names)
	data = df[["f1", "f2", "f3", "f4"]]
	labels = df["label"]

	data = np.array(data)
	labels = np.array(labels)

	scaler = MinMaxScaler()
	data = scaler.fit_transform(data)
	data = data.astype("float")

	label_encoder = LabelEncoder()
	labels = label_encoder.fit_transform(labels)
	labels = labels.astype("int")

	total_size = data.shape[0]
	random_permutation = np.random.permutation(np.arange(total_size))
	data = data[random_permutation]
	labels = labels[random_permutation]

	return data, labels

def get_Iris(batch_size=64, option_name = ''):
	data, labels = get_iris_np()

	# Convert to tensor dataset
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

	return dataloader, data_shape, data, labels

def get_ecoil_np():
	column_names = ["id", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "label"]

	df = pd.read_csv(folder_path + "Ecoli/ecoli.data", delimiter="\s+", header=None, names=column_names)
	data = df[["f1", "f2", "f3", "f4", "f5", "f6", "f7"]]
	labels = df["label"]

	data = np.array(data)
	labels = np.array(labels)
	label_encoder = LabelEncoder()
	labels = label_encoder.fit_transform(labels)
	labels = labels.astype("int")

	total_size = data.shape[0]
	random_permutation = np.random.permutation(np.arange(total_size))
	data = data[random_permutation]
	labels = labels[random_permutation]

	return data, labels

def get_Ecoil(batch_size=64, option_name = ''):
	data, labels = get_ecoil_np()
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)
	return dataloader, data_shape, data, labels

def get_dermatology_np():
	na_column = 33
	labels_column = 34
	df = pd.read_csv(folder_path + "Dermatology/dermatology.data", delimiter=",", header=None, na_values="?")
	mean_value = df[na_column].mean()
	df[na_column].fillna(value=mean_value, inplace=True)

	labels = df[labels_column]
	data = df.drop(columns=[labels_column])
	data = np.array(data)

	scaler = MinMaxScaler()
	data = scaler.fit_transform(data)
	data = data.astype("float")

	labels = np.array(labels)
	label_encoder = LabelEncoder()
	labels = label_encoder.fit_transform(labels)
	labels = labels.astype("int")
	total_size = data.shape[0]
	random_permutation = np.random.permutation(np.arange(total_size))
	data = data[random_permutation]
	labels = labels[random_permutation]
	return data, labels

def load_Dermatology(batch_size=64, option_name = ''):
	data, labels = get_dermatology_np()
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)
	return dataloader, data_shape, data, labels

def get_TCGA_np():
	df_data = pd.read_csv(folder_path + 'TCGA/data.csv', index_col=0)
	df_labels = pd.read_csv(folder_path + 'TCGA/labels.csv', index_col=0)

	data = np.array(df_data)
	labels = np.squeeze(np.array(df_labels))

	data = MinMaxScaler().fit_transform(data)
	labels = LabelEncoder().fit_transform(labels)

	return data, labels

def load_TCGA(batch_size=64, option_name = ''):
	data_np, labels_np = get_TCGA_np()
	# Convert to tensor dataset
	data_tr = torch.Tensor(data_np)
	data_shape = data_tr.shape[1]
	labels_tr = torch.Tensor(labels_np)
	final_dataset = TensorDataset(data_tr, labels_tr)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

	return dataloader, data_shape, data_np, labels_np

def get_newsgroups_dataloader(batch_size=64, data_points=-1):
	#pdb.set_trace()
	vectorizer = TfidfVectorizer()
	newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
	data = vectorizer.fit_transform(newsgroups_train.data)
	labels = newsgroups_train.target

	# Random Shuffle
	total_size = data.shape[0]
	data = data.todense()
	random_permutation = np.random.permutation(np.arange(total_size))
	data = data[random_permutation]
	labels = labels[random_permutation]

	# Select datapoints
	if (data_points > 0):
		data = data[:data_points]
		labels = labels[:data_points]

	for i in data[0]:
		print(i)

	# Convert to tensor dataset
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	del data, labels
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

	return dataloader, data_shape

def get_Pendigits_np():
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

def load_Pendigits(batch_size=64, option_name = ''):
	data_np, labels_np = get_Pendigits_np()
	# Convert to tensor dataset
	data_tr = torch.Tensor(data_np)
	data_shape = data_tr.shape[1]
	labels_tr = torch.Tensor(labels_np)
	final_dataset = TensorDataset(data_tr, labels_tr)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

	return dataloader, data_shape, data_np, labels_np

def load_coil20_dataloader(batch_size=64, option_name = ''):
	data_np = np.load(folder_path + "COIL-20/data.npy")
	labels_np = np.load(folder_path + "COIL-20/labels.npy")

	data_tr = torch.Tensor(data_np)
	data_tr = torch.reshape(data_tr, (data_tr.shape[0], 1, data_tr.shape[1], data_tr.shape[2]))
	data_shape = data_tr.shape[1]
	labels_tr = torch.Tensor(labels_np)
	final_dataset = TensorDataset(data_tr, labels_tr)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

	return dataloader, data_shape, data_np, labels_np

def load_10x73k_np():
	# Original Labels are within 0 to 9. But proper label mapping is required as there are 8 classes.
	data_path = folder_path + "10x_73k/sub_set-720.mtx"
	labels_path = folder_path + "10x_73k/labels.txt"

	# Read data
	data = scipy.io.mmread(data_path)
	data = data.toarray()
	data = np.float32(data)

	# Read labels
	labels = np.loadtxt(labels_path).astype(int)
	labels = LabelEncoder().fit_transform(labels)

	# Scale data
	data = np.log2(data + 1)
	scale = np.max(data)
	data = data / scale

	# New scaling
	data = MinMaxScaler().fit_transform(data)

	# TODO for conv
	#data = np.reshape(data, (-1, 1, data.shape[1]))

	return data, labels

def load_10x73k(batch_size=64, option_name=''):
	data_np, labels_np = load_10x73k_np()
	# Convert to tensor dataset
	data_tr = torch.Tensor(data_np)
	print(data_tr.shape)
	data_shape = data_tr.shape[1]
	print(data_shape)
	labels_tr = torch.Tensor(labels_np)
	final_dataset = TensorDataset(data_tr, labels_tr)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

	return dataloader, data_shape, data_np, labels_np

def get_fashionMNIST_np():
	IMG_SIZE = 28

	transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
	train = datasets.FashionMNIST(folder_path + "FashionMNIST", train=True, download=True, transform=transform)
	trainset = torch.utils.data.DataLoader(train, batch_size=1, shuffle=SHUFFLE)
	data = []
	labels = []

	for datapoint in enumerate(trainset):
		batch_idx, (example_data, example_targets) = datapoint
		example_targets = example_targets.item()

		example_data = example_data.view(1, IMG_SIZE, IMG_SIZE)
		data.append(np.array(example_data))
		labels.append(example_targets)
	np.save(folder_path + "FashionMNIST/FashionMNIST_subset/FashionMNIST.npy", data)
	np.save(folder_path + "FashionMNIST/FashionMNIST_subset/FashionMNIST_labels.npy", labels)

	return data, labels

def load_fashionMNIST_dataloader(batch_size=64, option_name=''):
	IMG_SIZE = 28
	data_np, labels_np = get_fashionMNIST_np()

	# Convert to tensor dataset
	data_tr = torch.Tensor(data_np)
	labels_tr = torch.Tensor(labels_np)
	final_dataset = TensorDataset(data_tr, labels_tr)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)
	
	return dataloader, IMG_SIZE * IMG_SIZE, data_np, labels_np

def get_eMNIST_balanced_np(option):
	IMG_SIZE = 28
	# Options: balanced, byclass, bymerge, digits, letters, mnist
	mndata = MNIST(folder_path + 'EMNIST')
	mndata.select_emnist('balanced')
	data, labels = mndata.load_training()
	data_ts, labels_ts = mndata.load_testing()

	data = np.vstack((data, data_ts))
	labels = np.hstack((labels, labels_ts))
	data = MinMaxScaler().fit_transform(data).astype(np.float32)

	data = np.reshape(data, (-1, 1, IMG_SIZE, IMG_SIZE))
	labels = np.array(labels)
	if option=='digits':
		valid_indices = np.where(labels < 10)[0]
	elif option=='letters':
		valid_indices = np.where((labels >= 10) & (labels<20))[0]
    
    
	data = data[valid_indices] 
	labels = labels[valid_indices]
	
	labels = LabelEncoder().fit_transform(labels)
	
	return data, labels

def get_eMNIST_mnist_np():
	IMG_SIZE = 28
	# Options: balanced, byclass, bymerge, digits, letters, mnist
	mndata = MNIST(folder_path + 'EMNIST')
	mndata.select_emnist('mnist')
	data, labels = mndata.load_training()
	data_ts, labels_ts = mndata.load_testing()

	data = np.vstack((data, data_ts))
	labels = np.hstack((labels, labels_ts))
	data = MinMaxScaler().fit_transform(data).astype(np.float32)

	data = np.reshape(data, (-1, 1, IMG_SIZE, IMG_SIZE))
	labels = np.array(labels)
	labels = LabelEncoder().fit_transform(labels)
	
	return data, labels


def load_eMNIST_dataloader(batch_size=64, option_name=''):
	IMG_SIZE = 28

	if option_name=='balanced letters':
		data_np, labels_np = get_eMNIST_balanced_np(option='letters')
	elif option_name=='mnist':
	    data_np, labels_np = get_eMNIST_mnist_np()
	elif option_name=='balanced digits':
		data_np, labels_np = get_eMNIST_balanced_np(option='digits')
		
	data_np = np.reshape(data_np,(data_np.shape[0], IMG_SIZE * IMG_SIZE))
	
    # Convert to tensor dataset
	data_tr = torch.Tensor(data_np)
	labels_tr = torch.Tensor(labels_np)
	final_dataset = TensorDataset(data_tr, labels_tr)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

	return dataloader, IMG_SIZE * IMG_SIZE, data_np, labels_np

def load_coil20_dataloader(batch_size=64, option_name=''):
    IMG_SIZE = 28
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),  # Convert PIL image to tensor
        # Add more transformations as needed
    ])
    dataset = datasets.ImageFolder(root=folder_path+'COIL-20/coil-20-proc', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=SHUFFLE)
    
    return dataloader, IMG_SIZE * IMG_SIZE, [], []     

def load_cifar10_dataloader(batch_size=64, option_name=''):
    IMG_SIZE = 28
    # Define the data transformations
    transform = transforms.Compose(
        [
         transforms.Resize((IMG_SIZE, IMG_SIZE)),
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor(),
         transforms.Normalize(0.5,0.5)])

    # Load the CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(root=folder_path+'CIFAR-10/data', train=True,
                                            download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=SHUFFLE, num_workers=2)
    
    return dataloader, IMG_SIZE * IMG_SIZE, [], []     

def load_stl10_dataloader(batch_size=64, option_name=''):
    IMG_SIZE = 28
    # Define the data transformations
    transform = transforms.Compose(
        [
         transforms.Resize((IMG_SIZE, IMG_SIZE)),
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0.5), std=(0.5))])

    # Load the STL-10 dataset
    dataset = torchvision.datasets.STL10(root=folder_path+'STL-10/data', split='train',
                                      download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=SHUFFLE, num_workers=2)
    return dataloader, IMG_SIZE * IMG_SIZE, [], []     

def normalize_data(X):
    # Calculate the L2 norm of X
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    
    # Avoid division by zero by handling zero norms
    norm_X[norm_X == 0] = 1.0# Normalize X
    normalized_X = X / norm_X

    return normalized_X

def load_R15_np():
    path = folder_path+"R15/R15.arff"
    df = pd.read_csv(path, skiprows=10, names=['x', 'y', 'class'])
    labels = df["class"].to_numpy()
    labels = LabelEncoder().fit_transform(labels)
    df.drop(columns=["class"], inplace=True)
    data = df.to_numpy()
    data = MinMaxScaler().fit_transform(data).astype(np.float32)
    # data = normalize_data(data)
    #############################   
    #valid_indices = np.where((labels == 0) | (labels == 9) | (labels == 14))[0]
    #data = data[valid_indices] 
    #labels = labels[valid_indices]
    #############################    
    return data, labels

def load_R15_dataloader(batch_size=64, option_name=''):
	data_np, labels_np = load_R15_np()
	# Convert to tensor dataset
	data_tr = torch.Tensor(data_np)
	data_shape = data_tr.shape[1]
	labels_tr = torch.Tensor(labels_np)
	final_dataset = TensorDataset(data_tr, labels_tr)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

	return dataloader, data_shape, data_np, labels_np

def load_R3_np():
    # Define the means and covariances for the Gaussian blobs
    means = np.array([[0, 0], [5, 0], [10, 0]])
    sigma = 0.1
    
    covariances = np.array([[[sigma, 0], 
                            [0, sigma]],
                            [[sigma, 0], 
                            [0, sigma]],
                            [[sigma, 0], 
                            [0, sigma]]])
    
    # Number of samples in each cluster
    n_samples = 500
    
    # Generate data for each cluster
    data = []
    for i in range(len(means)):
        cluster_data = np.random.multivariate_normal(means[i], covariances[i], n_samples)
        data.append(cluster_data)
    
    # Combine data from all clusters
    data = np.vstack(data)
    labels = np.repeat(np.arange(len(means)), n_samples)
    data = MinMaxScaler().fit_transform(data).astype(np.float32)
    # data = normalize_data(data)
    return data, labels

def load_R3_dataloader(batch_size=64, option_name=''):
	data_np, labels_np = load_R3_np()
	# Convert to tensor dataset
	data_tr = torch.Tensor(data_np)
	data_shape = data_tr.shape[1]
	labels_tr = torch.Tensor(labels_np)
	final_dataset = TensorDataset(data_tr, labels_tr)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

	return dataloader, data_shape, data_np, labels_np
