import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torch.optim as optim

import os

from SoftSilhouette import SoftSilhouette
from Visualization import Visualization
from Evaluations.Evaluation import Evaluator


class MLP(nn.Module):
	def __init__(self, device, n_clusters, input_dim):
		super(MLP, self).__init__()	
		self.device = device
		self.input_dim = input_dim
		self.n_clusters = n_clusters
		self.evaluator = Evaluator()
		self.softSilhouette = SoftSilhouette(self.device)

		self.model = nn.Sequential(
			nn.Linear(input_dim, n_clusters, bias=False),
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

		for weights in self.model[0].parameters():
			for index, center in enumerate(kmeans.cluster_centers_):
				with torch.no_grad(): # TODO
					center = torch.from_numpy(center) 
					center.requires_grad_()
					weights.data[index] = center
   
	def get_clustering_layer_centers(self):
		for weights in self.model[0].parameters():
			#for index in range(self.n_clusters):
				#print(weights.data[index]) 
			#print(weights)
			return weights

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
			self.data_list = list()
			self.clusters_list = list()
			self.labels_list = list()
			
			sum_soft_silhouette = 0
			sum_clustering_loss = 0
			sum_entropy = 0
			total_loss = 0

			for batch_index, (data, labels) in enumerate(self.dataloader):
				data = data.to(self.device)
				self.soft_clustering = self.forward(data).to(self.device)

				soft_sil = self.softSilhouette.soft_silhouette(data, self.soft_clustering, requires_distance_grad=True)
				clustering_loss = 1 - soft_sil
				entropy_val = entropy(np.transpose(self.soft_clustering.cpu().detach().numpy()), base=2) 
				entropy_val = self.entr_lambda * np.mean(entropy_val)
				total_loss = clustering_loss + entropy_val

				sum_soft_silhouette += soft_sil.item()
				sum_clustering_loss += clustering_loss.item()
				sum_entropy += entropy_val

				optimizer.zero_grad()
				total_loss.backward()
				optimizer.step()

				self.data_list.append(data.cpu().detach().numpy())
				self.labels_list.append(labels.cpu().detach().numpy())
				self.clusters_list.append(self.soft_clustering)

			self.clusters_list = torch.cat(self.clusters_list, dim=0)
			self.clusters_list = self.torch_to_numpy(self.clusters_list)
			self.labels_list = np.concatenate(self.labels_list).astype(int)
			self.data_list = np.concatenate(self.data_list)
			acc, pur, nmi, ari, sil = self.evaluator.evaluate_model(self.data_list, self.labels_list, self.clusters_list)
			self.df_eval.loc[epoch] = [sum_clustering_loss, sum_soft_silhouette, acc, pur, nmi, ari]

			print(f'Epoch: {epoch} Cl_loss: {sum_clustering_loss:.4f} Entropy: {sum_entropy:.4f} Soft_sil: {sum_soft_silhouette:.4f} SIL: {sil:.4f} ACC: {acc:.2f} PUR: {pur:.2f} NMI: {nmi:.2f} ARI: {ari:.2f}')
			
	def set_path(self):
		self.dest_path = os.path.join(self.dataset_name, '_With_', str(self.n_epochs), '_Eps_out_', str(self.n_clusters), '_bs_', str(self.batch_size), '_lr_', str(self.lr))
		self.data_dir_path = self.path_to_module + '/' + self.dataset_name + '/MLP/' + self.dest_path
			

	

		
		
		
		
