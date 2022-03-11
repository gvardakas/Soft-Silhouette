import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.io
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.datasets import fetch_20newsgroups, make_circles, fetch_olivetti_faces
from sklearn.feature_extraction.text import TfidfVectorizer
#from datapackage import Package
import pdb

# Do not change drop_last and shuffle!
drop_last = True
shuffle = False
IMG_SIZE = 28

def get_dermatology_np():
	na_column = 33
	labels_column = 34
	df = pd.read_csv("./datasets/Dermatology/dermatology.data", delimiter=",", header=None, na_values="?")
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

def get_dermatology_dataloader(batch_size=64):
	data, labels = get_dermatology_np()
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
	return dataloader, data_shape

def get_ecoil_np():
	column_names = ["id", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "label"]

	df = pd.read_csv("./datasets/Ecoli/ecoli.data", delimiter="\s+", header=None, names=column_names)
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

def get_ecoil_dataloader(batch_size=64):
	data, labels = get_ecoil_np()
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
	return dataloader, data_shape

def get_olivetti_faces_np():
	data_home = "./datasets/Olivetti-Faces"
	data, labels = fetch_olivetti_faces(data_home=data_home, return_X_y=True, shuffle=True)
	return data, labels

def get_olivetti_faces_dataloader(batch_size=64):
	data, labels = get_olivetti_faces_np()
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
	return dataloader, data_shape

def get_australian_np():
	column_names = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "label"]

	df = pd.read_csv("./datasets/Australian/australian.dat", delimiter=" ", header=None, names=column_names)
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

def get_australian_dataloader(batch_size=69):
	data, labels = get_australian_np()

	# Convert to tensor dataset
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)

	return dataloader, data_shape

def get_wine_np():
	column_names = ["label", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13"]
	df = pd.read_csv("./Wine/wine.data", header=None, names=column_names)
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

def get_wine_dataloader(batch_size=89):
	data, labels = get_wine_np()

	# Convert to tensor dataset
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)

	return dataloader, data_shape

def get_ring_np():
	data, labels = make_circles(n_samples=1_000, factor=0.3, noise=0.05, random_state=0)
	scaler = MinMaxScaler()
	data = scaler.fit_transform(data)
	data = data.astype("float")
	labels = labels.astype("int")
	return data, labels

def get_rings_dataloader(batch_size=64):
	data, labels = get_ring_np()

	# Convert to tensor dataset
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)

	return dataloader, data_shape

def get_iris_np():
	columnn_names = ["f1", "f2", "f3", "f4", "label"]
	df = pd.read_csv("./datasets/Iris/iris.data", header=None, names=columnn_names)
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

def get_iris_dataloader(batch_size=50):
	data, labels = get_iris_np()

	# Convert to tensor dataset
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)

	return dataloader, data_shape


def get_synthetic_np():
	data = np.load("./datasets/Synthetic/synthetic.npy")
	labels = np.load("./datasets/Synthetic/synthetic_labels.npy")
	total_size = data.shape[0]

	random_permutation = np.random.permutation(np.arange(total_size))
	data = data[random_permutation]
	labels = labels[random_permutation]

	return data, labels

def get_synthetic_dataloader(batch_size=64):
	data, labels = get_synthetic_np()

	# Convert to tensor dataset
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)

	return dataloader, data_shape

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
	dataloader = DataLoader(final_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)

	return dataloader, data_shape

def get_10x_73k_np(data_points=-1):
	# Original Labels are within 0 to 9.
	# But proper label mapping is required as there are 8 classes.
	map_labels = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5, 8: 6, 9: 7}
	data_path = "./datasets/10x_73k/sub_set-720.mtx"
	labels_path = "./datasets/10x_73k/labels.txt"

	# Read data
	data = scipy.io.mmread(data_path)
	data = data.toarray()
	data = np.log2(data + 1)
	scale = np.max(data)
	data = data / scale
	total_size = data.shape[0]
	random_permutation = np.random.permutation(np.arange(total_size))
	data = data[random_permutation]

	# Read labels
	labels = np.loadtxt(labels_path).astype(int)
	labels = labels[random_permutation]
	labels = np.array([map_labels[i] for i in labels])

	if (data_points > 0):
		data = data[:data_points]
		labels = labels[:data_points]

	return data, labels

def get_10x_73k(batch_size=64, data_points=-1):
	data, labels = get_10x_73k_np(data_points=data_points)
	# Convert to tensor dataset
	data = torch.Tensor(data)
	data_shape = data.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)

	return dataloader, data_shape

def get_MNIST_dataloader(batch_size=64):
	transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
	train = datasets.MNIST("./datasets/MNIST", train=True, download=True, transform=transform)
	dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
	return dataloader

def get_FashionMNIST_dataloader(batch_size=64):
	transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
	train = datasets.FashionMNIST("./datasets/FashionMNIST", train=True, download=True, transform=transform)
	dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
	return dataloader

def get_FashionMNIST_5_dataloader(batch_size=64):
	data = np.load("./datasets/FashionMNIST_5/fashionmnist_5.npy")
	labels = np.load("./datasets/FashionMNIST_5/fashionmnist_5_labels.npy")

	# Convert to tensor dataset
	data = torch.Tensor(data)
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	del data, labels

	dataloader = DataLoader(final_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
	return dataloader

def get_Pendigits_np():
	data = np.loadtxt("./datasets/Pendigits/pendigits.tra.txt", delimiter=',')
	pendigits, labels = data[:, :-1], data[:, -1:]

	# Fix pendigits
	pendigits = pendigits.astype("float")
	scaler = MinMaxScaler()
	pendigits = scaler.fit_transform(pendigits)

	# Fix labels
	labels = np.squeeze(labels)
	labels = labels.astype("int")

	return pendigits, labels

def get_Pendigits_dataloader(batch_size=64):
	pendigits, labels = get_Pendigits_np()
	# Construct the dataset and the dataloader
	pendigits = torch.Tensor(pendigits)
	data_shape = pendigits.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(pendigits, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)

	return dataloader, data_shape

def get_even_or_odd_Pendigits_dataloader(batch_size=64, even=True):
	package = Package('https://datahub.io/machine-learning/pendigits/datapackage.json')
	data = package.get_resource("pendigits_csv").read()
	data = np.array(data)

	pendigits, labels = data[:, :-1], data[:, -1:]
	del data

	# Fix pendigits
	pendigits = pendigits.astype("float")
	scaler = MinMaxScaler()
	pendigits = scaler.fit_transform(pendigits)

	# Fix labels
	labels = np.squeeze(labels)
	labels = labels.astype("int")

	# Select even numbers
	if even:
		indexes = np.where(labels % 2 == 0)
	else:
		indexes = np.where(labels % 2 != 0)

	pendigits = pendigits[indexes]
	labels = labels[indexes]

	# Construct the dataset and the dataloader
	pendigits = torch.Tensor(pendigits)
	data_shape = pendigits.shape[1]
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(pendigits, labels)
	del pendigits, labels
	dataloader = DataLoader(final_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)

	return dataloader, data_shape

def get_MNIST_subset_np(create_subset=False, data_per_pattern=1000):
	if(create_subset):
		transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
		train = datasets.MNIST("./datasets/MNIST", train=True, download=True, transform=transform)
		trainset = torch.utils.data.DataLoader(train, batch_size=1, shuffle=shuffle)
		data = []
		labels = []
		dict_counter = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
		data_per_pattern = data_per_pattern

		for datapoint in enumerate(trainset):
			current_size = sum(dict_counter.values())
			target_size = len(dict_counter) * data_per_pattern

			if current_size >= target_size: break
			batch_idx, (example_data, example_targets) = datapoint
			example_targets = example_targets.item()

			if example_targets in dict_counter and dict_counter[example_targets] < data_per_pattern:
				dict_counter[example_targets] += 1
				example_data = example_data.view(1, IMG_SIZE, IMG_SIZE)
				data.append(np.array(example_data))
				labels.append(example_targets)

		np.save("datasets/MNIST/MNIST_subset/MNIST.npy", data)
		np.save("datasets/MNIST/MNIST_subset/MNIST_labels.npy", labels)

	# Load data
	data = np.load("datasets/MNIST/MNIST_subset/MNIST.npy")
	labels = np.load("datasets/MNIST/MNIST_subset/MNIST_labels.npy")
	
	return data, labels

def get_MNIST_subset_dataloader(batch_size=64, create_subset=False, data_per_pattern=1000):
	data, labels = get_MNIST_subset_np(create_subset, data_per_pattern)

	# Convert to tensor dataset
	data = torch.Tensor(data)
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
	return dataloader

def get_fashionMNIST_np(create_subset=False, data_per_pattern=1000):
	if(create_subset):
		transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
		train = datasets.FashionMNIST("./datasets/FashionMNIST", train=True, download=True, transform=transform)
		trainset = torch.utils.data.DataLoader(train, batch_size=1, shuffle=shuffle)
		data = []
		labels = []
		dict_counter = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
		data_per_pattern = data_per_pattern

		for datapoint in enumerate(trainset):
			current_size = sum(dict_counter.values())
			target_size = len(dict_counter) * data_per_pattern

			if current_size >= target_size: break
			batch_idx, (example_data, example_targets) = datapoint
			example_targets = example_targets.item()

			if example_targets in dict_counter and dict_counter[example_targets] < data_per_pattern:
				dict_counter[example_targets] += 1
				example_data = example_data.view(1, IMG_SIZE, IMG_SIZE)
				data.append(np.array(example_data))
				labels.append(example_targets)
		np.save("datasets/FashionMNIST/FashionMNIST_subset/FashionMNIST.npy", data)
		np.save("datasets/FashionMNIST/FashionMNIST_subset/FashionMNIST_labels.npy", labels)

	# Load data
	data = np.load("datasets/FashionMNIST/FashionMNIST_subset/FashionMNIST.npy")
	labels = np.load("datasets/FashionMNIST/FashionMNIST_subset/FashionMNIST_labels.npy")

	return data, labels


def get_fashionMNIST_subset_dataloader(batch_size=64, create_subset=False, data_per_pattern=1000):
	data, labels = get_fashionMNIST_np(create_subset, data_per_pattern)

	# Convert to tensor dataset
	data = torch.Tensor(data)
	labels = torch.Tensor(labels)
	final_dataset = TensorDataset(data, labels)
	dataloader = DataLoader(final_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
	return dataloader
