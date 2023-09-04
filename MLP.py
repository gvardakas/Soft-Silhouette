import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.nn.functional import softmax, normalize
import torch.optim as optim

import os

from Objectives import Objectives
from Visualization import Visualization
from Evaluations.Evaluation import Evaluator

class MLP(nn.Module):
    
    def __init__(self, device, n_clusters, input_dim):
        super(MLP, self).__init__() 
        self.device = device
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.evaluator = Evaluator()
        self.objectives = Objectives(self.device)

        self.model = nn.Sequential(
            nn.Linear(input_dim, n_clusters, bias=True), # TODO Look this
            #nn.ReLU(inplace=True),
            #nn.Sigmoid(),
            #nn.BatchNorm1d(n_clusters)
        )

    def set_training_variables(self, dataloader, batch_size, n_epochs, lr, entr_lambda):
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.entr_lambda = entr_lambda

    def set_path_variables(self, path_to_module, dataset_name):
        self.path_to_module = path_to_module
        self.dataset_name = dataset_name

    def forward(self, x):
        x = x.to(self.device)
        x = self.model(x)
        x = softmax(x, dim=1)
        return x
    
    def get_data(self):
        data_list, labels_list = list(), list()

        for batch_index, (data, labels) in enumerate(self.dataloader):
            data_list.append(data)
            labels_list.append(labels)

        return np.concatenate(data_list), np.concatenate(labels_list).astype(int)
    
    def kmeans_initialization(self, n_init=10):
        data, labels = self.get_data()    
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=n_init).fit(data)
        init_center = torch.from_numpy(kmeans.cluster_centers_).to(self.device)
        self.model[0].weight = nn.Parameter(init_center)
   
    def get_clustering_layer_centers(self):
        return self.model[0].weight

    def torch_to_numpy(self, clusters):
        # Get the data clusters based on max neuron
        clusters = torch.argmax(clusters, dim=1)
        clusters = clusters.cpu().detach().numpy()
        return clusters

    def train(self):
        self.df_eval = pd.DataFrame(columns=['Cl_Loss','Soft_Sil','Accuracy','Purity','Nmi','Ari'])
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        # Train MLP
        for epoch in range(self.n_epochs):
            sum_soft_silhouette = 0
            sum_clustering_loss = 0
            sum_entropy = 0
            total_loss = 0

            self.data_list = list()
            self.clusters_list = list()
            self.labels_list = list()

            for batch_index, (data, labels) in enumerate(self.dataloader):
                self.soft_clustering = self.forward(data).to(self.device)

                soft_sil = self.objectives.soft_silhouette(data, self.soft_clustering, requires_distance_grad=True)
                mean_entropy, _ = self.objectives.entropy(self.soft_clustering, base=2)

                clustering_loss = 1 - soft_sil
                total_loss = clustering_loss + self.entr_lambda * mean_entropy

                sum_soft_silhouette += soft_sil.item()
                sum_clustering_loss += clustering_loss.item()
                sum_entropy += self.entr_lambda * mean_entropy.item()

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                self.data_list.append(data)
                self.labels_list.append(labels)
                self.clusters_list.append(self.soft_clustering)

            self.clusters_list = torch.cat(self.clusters_list, dim=0)
            self.clusters_list = self.torch_to_numpy(self.clusters_list)
            self.labels_list = np.concatenate(self.labels_list).astype(int)
            self.data_list = np.concatenate(self.data_list)
            acc, pur, nmi, ari, sil = self.evaluator.evaluate_model(self.data_list, self.labels_list, self.clusters_list)
            self.df_eval.loc[epoch] = [sum_clustering_loss, sum_soft_silhouette, acc, pur, nmi, ari]

            print(f'Epoch: {epoch} Cl Loss: {sum_clustering_loss:.4f} Entropy: {sum_entropy:.4f} Soft Sil: {sum_soft_silhouette:.4f} SIL: {sil:.4f} ACC: {acc:.2f} PUR: {pur:.2f} NMI: {nmi:.2f} ARI: {ari:.2f}')
        
        return self.data_list, self.labels_list, self.clusters_list

    def set_path(self):
        self.properties_name = str(self.n_epochs) + '_Eps'
        self.properties_name += '_out_' + str(self.n_clusters)
        self.properties_name += '_bs_' + str(self.batch_size) + '_lr_' + str(self.lr)
        self.properties_name += '_entr_lambda_' + str(self.entr_lambda)
        self.data_dir_path = self.path_to_module + 'Results/' + self.dataset_name + '/MLP/' + self.properties_name
        
        
        
        
