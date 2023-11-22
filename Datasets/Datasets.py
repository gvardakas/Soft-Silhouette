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
from sklearn.datasets import make_circles, make_moons, make_blobs, fetch_20newsgroups, fetch_openml
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
from sklearn.datasets import fetch_rcv1

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, reuters

# Download the missing tokenizer data
nltk.download('punkt')
nltk.download('stopwords')

SHUFFLE = True 
IMG_SIZE = 28

folder_path = './Datasets/'

def get_3d_spheres_np():
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

def get_moons_np():
    data, labels = make_moons(n_samples=1_000, noise=0.05, random_state=0)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = normalize_data(data)
    data = data.astype("float")
    labels = labels.astype("int")
    return data, labels
    
def get_squeezed_gauss_np():
    # Do not normilize! #2
    data, labels = make_blobs(n_samples=1000,random_state=0, centers=[(0.45, 0.5), (0.55, 0.5)], cluster_std=((0.005, 0.15), (0.005, 0.15)))
    data = data.astype("float")
    labels = labels.astype("int")
    return data, labels
    
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

def get_ring_np():
    data, labels = make_circles(n_samples=1_000, factor=0.3, noise=0.05, random_state=0)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = data.astype("float")
    labels = labels.astype("int")
    return data, labels

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
"""
def get_tcga_np():
    df_data = pd.read_csv(folder_path + 'TCGA/data.csv', index_col=0)
    df_labels = pd.read_csv(folder_path + 'TCGA/labels.csv', index_col=0)

    data = np.array(df_data)
    labels = np.squeeze(np.array(df_labels))

    data = MinMaxScaler().fit_transform(data)
    labels = LabelEncoder().fit_transform(labels)

    return data, labels
"""
def get_tcga_np():
    df_train = pd.read_csv(folder_path + 'HARS/train.csv', index_col=0)

    df_test = pd.read_csv(folder_path + 'HARS/test.csv', index_col=0)

    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    del df_train, df_test
    df_labels = df['Activity']
    df.drop(columns=['subject','Activity'], inplace=True)
    data = np.array(df)
    labels = np.squeeze(np.array(df_labels))

    data = MinMaxScaler().fit_transform(data)   
    labels = LabelEncoder().fit_transform(labels)

    return data, labels    

from nltk.corpus import reuters, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

n_classes = 90
labels = reuters.categories()


def load_data(config={}):
    """
    Load the Reuters dataset.

    Returns
    -------
    data : dict
        with keys 'x_train', 'x_test', 'y_train', 'y_test', 'labels'
    """
    stop_words = stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    label_encoder = LabelEncoder()

    documents = reuters.fileids()
    test = [d for d in documents if d.startswith('test/')]
    train = [d for d in documents if d.startswith('training/')]

    docs = {}
    docs['train'] = [reuters.raw(doc_id) for doc_id in train]
    xs = {'train': []}
    xs['train'] = vectorizer.fit_transform(docs['train']).toarray()
    ys = {'train': []}
    ys['train'] = label_encoder.fit_transform(np.array([reuters.categories(doc_id)
                                     for doc_id in train]))
    data = {'x_train': xs['train'], 'y_train': ys['train'],
            'labels': globals()["labels"]}
    return data


def get_reuters_4_np():
    # Load the RCV1 dataset
    rcv1 = fetch_rcv1(subset='all')
    
    # Determine the indices of documents with multiple categories
    docs_with_multiple_categories = []
    for i, target in enumerate(rcv1.target):
        if np.sum(target) > 1:
            docs_with_multiple_categories.append(i)
    
    # Remove documents with multiple categories
    filtered_indices = [i for i in range(rcv1.data.shape[0]) if i not in docs_with_multiple_categories]
    
    # Create a new dataset with single-category documents
    single_category_data = [rcv1.data[i] for i in filtered_indices]
    single_category_target = [rcv1.target[i] for i in filtered_indices]
    
    # Filter the data based on your selected categories
    selected_categories = ["CCAT", "ECAT", "GCAT", "MCAT", "GCAT", "GCAT"]
    filtered_indices = []
    for i, target in enumerate(rcv1.target):
        if any(category in target for category in selected_categories):
            filtered_indices.append(i)
    
    X = rcv1.data[filtered_indices]
    labels = rcv1.target[filtered_indices]
    print(labels.unique())
    # Preprocess the data (e.g., TF-IDF vectorization)
    tfidf = TfidfVectorizer(max_features=2000)
    X = tfidf.fit_transform(X)
    data = MinMaxScaler().fit_transform(X.toarray())
    # Label Encoding
    labels = LabelEncoder().fit_transform(labels)


    #############################   
    #valid_indices = np.where(labels < 5)[0]
    #data = data[valid_indices] 
    #labels = labels[valid_indices]
    #############################  
    return data, labels

def get_20_newsgroups_np():
    """
    # Load the dataset
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    # Clean the text
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = text.lower()
        return text

    cleaned_texts = [clean_text(text) for text in newsgroups.data]

    # Tokenization
    def tokenize_text(text):
        return word_tokenize(text)

    tokenized_texts = [tokenize_text(text) for text in cleaned_texts]

    # Remove stop words
    def remove_stopwords(tokens):
        stop_words = set(stopwords.words('english'))
        return [word for word in tokens if word not in stop_words]

    preprocessed_texts = [remove_stopwords(tokens) for tokens in tokenized_texts]
    """
    # remove the headers, footers, and quotes from the documents.
    dataset = fetch_20newsgroups(subset='all', shuffle=False, remove=('headers', 'footers', 'quotes'))
    print(dataset.data[0])
    
    corpus = dataset.data # save as the raw docs
    gnd_labels = dataset.target # labels for clustering evaluation or supervised tasks
    print(len(corpus), len(gnd_labels))
    print(type(corpus), type(gnd_labels))
    print(gnd_labels)
    print(dataset.target_names)
    
    # perform more Pre-processing steps
    from collections import defaultdict
    from nltk.tokenize import RegexpTokenizer
    from stop_words import get_stop_words
    from nltk.stem.porter import PorterStemmer
    from gensim import corpora
    from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
    from pprint import pprint

    def pre_processing(docs):
        tokenizer = RegexpTokenizer(r"\w+(?:[-'+]\w+)*|\w+")
        en_stop = get_stop_words('en')
        for doc in docs:
            raw_text = doc.lower()
            # tokenization
            tokens_text = tokenizer.tokenize(raw_text)
            # remove stopwords
            stopped_tokens_text = [i for i in tokens_text if not i in en_stop]
            # remoce digis and one-charcter word
            doc = [token for token in stopped_tokens_text if not token.isnumeric()]
            doc = [token for token in stopped_tokens_text if len(token) > 1]
            # you could always add some new preprocessing here
            yield doc
    # Preprocess all the documents in the corpus
    Vocab_v1 = list(pre_processing(corpus))
    # verify length of the clean corpus and print a sample clean tokenized document
    print(len(Vocab_v1))
    print(Vocab_v1[0])
        
    
    # Vectorization using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=2000)  # You can adjust the number of features as needed
    X = tfidf_vectorizer.fit_transform([' '.join(tokens) for tokens in Vocab_v1])

    data = MinMaxScaler().fit_transform(X.toarray())
    # Label Encoding
    labels = LabelEncoder().fit_transform(gnd_labels)


    #############################   
    #valid_indices = np.where(labels < 5)[0]
    #data = data[valid_indices] 
    #labels = labels[valid_indices]
    #############################  
    return data, labels

"""
def get_20_newsgroups_np():
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    # Preprocessing: TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=2000, stop_words='english')
    X = vectorizer.fit_transform(newsgroups.data)

    data = MinMaxScaler().fit_transform(X.toarray())
    labels = LabelEncoder().fit_transform(newsgroups.target)
    
    return data, labels 
"""


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
    #pendigits = normalize_data(pendigits)
    # Fix labels
    labels = np.squeeze(labels)
    labels = labels.astype("int")

    return pendigits, labels

def get_hars_np():
    df_train = pd.read_csv(folder_path + 'HARS/train.csv', index_col=0)
    df_test = pd.read_csv(folder_path + 'HARS/test.csv', index_col=0)
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    del df_train, df_test
    
    labels = df['Activity']
    df.drop(columns=['subject','Activity'], inplace=True)
    data = np.array(df)
    labels = np.squeeze(np.array(labels))

    data = MinMaxScaler().fit_transform(data)
    labels = LabelEncoder().fit_transform(labels)

    return data, labels    
    
def get_10x73k_np():
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
"""
def get_fashionmnist_np():

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
"""

def get_fashionmnist_np():
    
    # Tshirt/Top, Dress
    # Trouser
    # Pullover, Coat, Shirt
    # Bag
    # Sandal, Sneaker, Ankle Boot

    # Define the category mapping
    category_mapping = {
        0: 0,  
        1: 1,  
        2: 2,  
        3: 0,  
        4: 2,  
        5: 3,
        6: 2,
        7: 3,
        8: 4,
        9: 3
    }


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
        
        # Map the original category to the merged category
        mapped_category = category_mapping[example_targets]
        labels.append(mapped_category)

    np.save(folder_path + "FashionMNIST/FashionMNIST_subset/FashionMNIST.npy", data)
    np.save(folder_path + "FashionMNIST/FashionMNIST_subset/FashionMNIST_labels.npy", labels)

    return data, labels

def get_r15_np():
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

def get_r3_np():
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

def get_r100_np():
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
    mndata = MNIST(folder_path + 'EMNIST')
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

def get_emnist_balanced_letters_np():
    data, labels = get_emnist_general_np('letters')
    valid_indices = np.where((labels >=20 ) & (labels < 30))[0]
    
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

def get_usps_np():
    # Load the USPS dataset using Scikit-Learn
    usps = fetch_openml(name="usps", version=2)
    
    # Extract the data and labels as NumPy arrays
    usps_data = usps.data.astype(float).to_numpy()
    usps_labels = usps.target.astype(int).to_numpy()
    
    # Initialize an empty array for resized images (28x28) with padding filled with -1
    usps_data_padded = np.full((len(usps_data), 28, 28), -1.0)
    
    # Calculate padding size (6 pixels on each side)
    padding = (28 - 16) // 2
    
    # Loop through each image and add padding with -1 values to make it 28x28
    for i in range(len(usps_data)):
        # Reshape the image to 16x16 pixels
        image = usps_data[i].reshape(16, 16).astype('float32')
        
        # Create a 28x28 canvas filled with -1 values
        padded_image = np.full((28, 28), -1.0)
        
        # Paste the original image into the center of the canvas
        padded_image[padding:padding+16, padding:padding+16] = image # Scale pixel values to 0-1
        
        # Store the padded image in the new array
        usps_data_padded[i] = padded_image
        
        data = np.reshape(usps_data_padded, (-1, 1, 28, 28))
        print(data[0].shape)
        labels = LabelEncoder().fit_transform(usps_labels)
        
    return data, labels

def get_kmnist_np():
    
    # Load the data
    data = np.load(folder_path + 'KMNIST/kmnist-train-imgs.npz')['arr_0']
    labels = np.load(folder_path + 'KMNIST/kmnist-train-labels.npz')['arr_0']
    #data = data.astype('float32')
    #data /= 255.0
    data = data.reshape(-1, data.shape[-1])
    
    data = MinMaxScaler().fit_transform(data).astype(np.float32)

    data = np.reshape(data, (-1, 1, IMG_SIZE, IMG_SIZE))
    labels = np.array(labels)
    labels = LabelEncoder().fit_transform(labels)
    
    return data, labels

def normalize_data(X):
    # Calculate the L2 norm of X
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    
    # Avoid division by zero by handling zero norms
    norm_X[norm_X == 0] = 1.0# Normalize X
    normalized_X = X / norm_X

    return normalized_X

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
