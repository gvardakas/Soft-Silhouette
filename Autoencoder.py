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
from General_Functions import General_Functions
import torch_rbf as rbf

class GenericAutoencoder(nn.Module):

	def __init__(self, device, n_clusters, input_dim, latent_dim, negative_slope):
		super(GenericAutoencoder, self).__init__()
		self.IMG_SIZE = 28
		self.device = device
		self.n_clusters = n_clusters
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.negative_slope = negative_slope
		self.kernel = rbf.gaussian

		self.evaluator = Evaluator()
		self.objectives = Objectives(self.device)

	def set_general_training_variables(self, dataloader, batch_size):
		self.dataloader = dataloader
		self.batch_size = batch_size

	def set_pretraining_variables(self, n_pret_epochs, pret_lr):
		self.n_pret_epochs = n_pret_epochs
		self.pret_lr = pret_lr

	def set_training_variables(self, n_epochs, lr, sil_lambda, entr_lambda):
		self.n_epochs = n_epochs
		self.lr = lr
		self.sil_lambda = sil_lambda
		self.entr_lambda = entr_lambda

	def set_path_variables(self, path_to_module, dataset_name):
		self.path_to_module = path_to_module
		self.dataset_name = dataset_name
		
	# Forward to Encoder and Decoder Models
	def forward(self, x):
		x = x.to(self.device)
		x = self.encoder_model(x)
		x = self.decoder_model(x)
		return x

	# Forward for data x through Encoder Model
	def encoder(self, x):
		x = x.to(self.device)
		x = self.encoder_model(x)
		return x

	# Forward for data x through Decoder Model
	def decoder(self, x):
		x = x.to(self.device)
		x = self.decoder_model(x)
		return x

	# Forward with Soft Max for data x through Encoder Model (1st Part), MLP Model (2nd Part) and SoftMax Function (3rd Part)
	def forward_softMax(self, x, temp=0.25):
		x = x.to(self.device)
		x = self.encoder_model(x)
		x = self.cluster_model(x)
		x = softmax(x/temp, dim=1)
		return x

	def get_latent_data(self):
		latent_data_list, labels_list = list(), list()

		for batch_index, (data, labels) in enumerate(self.dataloader):
			if self.needsReshape: #TODO
				data = torch.reshape(data, (data.shape[0], self.n_channels, self.IMG_SIZE, self.IMG_SIZE))
				
			code = self.encoder(data).to(self.device)
			code = code.cpu().detach().numpy()

			latent_data_list.append(code)
			labels_list.append(labels)

		return np.concatenate(latent_data_list), np.concatenate(labels_list).astype(int)
	
	def kmeans_initialization(self, n_init=10):
		latent_data, labels = self.get_latent_data()
		kmeans = KMeans(n_clusters=self.n_clusters, n_init=n_init).fit(latent_data)
		init_centers = torch.from_numpy(kmeans.cluster_centers_).to(self.device)
		self.cluster_model[0].centres = nn.Parameter(init_centers)
		
		self.evaluator.evaluate_model(latent_data, labels, kmeans.labels_)
		self.evaluator.print_evaluation()
	
   
	def get_cluster_centers(self):
		return self.cluster_model[0].centres

	def save_pretrained_weights(self):
		# Set the file path where you want to save the model's state dictionary
		model_save_path = self.data_dir_path + "/Weigths/autoencoder_weights.pth"
		
		General_Functions().create_directory(self.data_dir_path + "/Weigths")
		
		# Save the model's state dictionary
		torch.save(self.state_dict(), model_save_path)

	def torch_to_numpy(self, clusters):
		# Get the data clusters based on max neuron
		clusters = torch.argmax(clusters, dim=1)
		clusters = clusters.cpu().detach().numpy()
		return clusters
		
	def pretrain_autoencoder(self):
		MSE = nn.MSELoss().to(self.device)
		optimizer = optim.Adam(self.parameters(), lr=self.pret_lr)

		for epoch in range(self.n_pret_epochs):
			sum_rec_loss = 0
			for batch_index, (data, labels) in enumerate(self.dataloader):
				data = data.to(self.device)

				if self.needsReshape: # TODO
					data = torch.reshape(data, (data.shape[0], self.n_channels, self.IMG_SIZE, self.IMG_SIZE))
				
				reconstructions = self.forward(data)
				rec_loss = MSE(reconstructions, data)

				sum_rec_loss += rec_loss.item()

				optimizer.zero_grad()
				rec_loss.backward()
				optimizer.step()
				
			print(f'Epoch: {epoch}, Loss: {sum_rec_loss:.6f}')

		return self
	
	def train_autoencoder(self):
		self.df_eval = pd.DataFrame(columns=['Rec_Loss','Cl_Loss','Soft_Sil','Accuracy','Purity','Nmi','Ari'])

		optimizer = optim.Adam(self.parameters(), lr=self.lr)
		MSE = nn.MSELoss()
		
		# Train Autoencoder
		for epoch in range(self.n_epochs):
			sum_rec_loss = 0
			sum_soft_silhouette = 0
			sum_clustering_loss = 0
			sum_entropy = 0

			self.clusters_list = list()
			self.labels_list = list()
			self.latent_data_list = list()
			
			for batch_index, (data, labels) in enumerate(self.dataloader):
				data = data.to(self.device)
	
				if self.needsReshape: #TODO
					data = torch.reshape(data, (data.shape[0], self.n_channels, self.IMG_SIZE, self.IMG_SIZE))
				
				reconstructions = self.forward(data)
				code = self.encoder(data)
				soft_clustering = self.forward_softMax(data).to(self.device)

				soft_sil = self.objectives.soft_silhouette(code, soft_clustering, requires_distance_grad=True)
				rec_loss = MSE(reconstructions, data)

				clustering_loss = 1 - soft_sil
				mean_entropy, _ = self.objectives.entropy(soft_clustering, base=2)

				total_loss = rec_loss + self.sil_lambda * clustering_loss + self.entr_lambda * mean_entropy

				sum_rec_loss += rec_loss.item()
				sum_clustering_loss += self.sil_lambda * clustering_loss.item()
				sum_entropy += self.entr_lambda * mean_entropy.item()
				sum_soft_silhouette += soft_sil.item()


				optimizer.zero_grad()
				total_loss.backward()
				optimizer.step()

				# Take argmax from soft clustering
				self.clusters_list.append(soft_clustering)
				self.labels_list.append(labels)
				self.latent_data_list.append(code.cpu().detach().numpy())
				
			self.clusters_list = torch.cat(self.clusters_list, dim=0)
			self.clusters_list = self.torch_to_numpy(self.clusters_list)
			self.labels_list = np.concatenate(self.labels_list).astype(int)
			self.latent_data_list = np.concatenate(self.latent_data_list)

			acc, pur, nmi, ari, sil = self.evaluator.evaluate_model(self.latent_data_list, self.labels_list, self.clusters_list)
			self.df_eval.loc[epoch] = [sum_rec_loss, sum_clustering_loss, sum_soft_silhouette, acc, pur, nmi, ari]
			print(f'Ep: {epoch} Rec L: {sum_rec_loss:.4f} Cl L: {sum_clustering_loss:.4f} Entropy: {sum_entropy:.4f} SSil: {sum_soft_silhouette:.4f} SIL: {sil:.4f} ACC: {acc:.2f} PUR: {pur:.2f} NMI: {nmi:.2f} ARI: {ari:.2f}')

		return self.latent_data_list, self.labels_list, self.clusters_list

	def set_path(self):
		self.properties_name = str(self.n_epochs) + '_Eps'
		self.properties_name += '_ld_' + str(self.latent_dim) + '_out_' + str(self.n_clusters)
		self.properties_name += '_bs_' + str(self.batch_size) + '_lr_' + str(self.lr)
		self.properties_name += '_sil_lambda_' + str(self.sil_lambda) + '_entr_lambda_' + str(self.entr_lambda)
		self.data_dir_path = self.path_to_module + 'Results/' + self.dataset_name + '/AE/' + self.properties_name
		
class Autoencoder(GenericAutoencoder):
	def __init__(self,  device, n_clusters, input_dim, latent_dim, negative_slope):
		super(Autoencoder, self).__init__( device, n_clusters, input_dim, latent_dim, negative_slope)
		self.needsReshape = False
				
		self.encoder_model = nn.Sequential(
			nn.Linear(self.input_dim, 500, bias = True),
			nn.LeakyReLU(negative_slope = self.negative_slope, inplace=True),
			nn.BatchNorm1d(500),

			nn.Linear(500, 500, bias = True),
			nn.LeakyReLU(negative_slope = self.negative_slope, inplace=True),
			nn.BatchNorm1d(500),

			nn.Linear(500, 2000, bias = True),
			nn.LeakyReLU(negative_slope = self.negative_slope, inplace=True),
			nn.BatchNorm1d(2000),

			nn.Linear(2000, self.latent_dim, bias = True),
			nn.Tanh(),
			nn.BatchNorm1d(self.latent_dim)
		)

		# Clustering MLP - MLP Part from latent Dimension to Number of Clusters
		self.cluster_model = nn.Sequential(
			
			# Output Layer
			# nn.Linear(self.latent_dim, self.n_clusters, bias=True), # TODO Look This
			rbf.RBF(self.latent_dim, self.n_clusters, self.kernel),

		)
	
		# Decoder Model - ([Latent Space, Linear], [2000, LeakyReLU], [500, LeakyReLU], [500, LeakyReLU], [Input Space, Linear])
		self.decoder_model = nn.Sequential(
			nn.Linear(self.latent_dim, 2000, bias = True),
			nn.LeakyReLU(negative_slope = self.negative_slope, inplace=True),
			nn.BatchNorm1d(2000),
	
			nn.Linear(2000, 500, bias = True),
			nn.LeakyReLU(negative_slope = self.negative_slope, inplace=True),
			nn.BatchNorm1d(500),
	
			nn.Linear(500, 500, bias = True),
			nn.LeakyReLU(negative_slope = self.negative_slope, inplace=True),
			nn.BatchNorm1d(500),
	
			nn.Linear(500, self.input_dim, bias = True),
			nn.LeakyReLU(negative_slope = self.negative_slope, inplace=True)
		)
		
class CD_Autoencoder(GenericAutoencoder):

	def __init__(self, device, n_clusters, input_dim, latent_dim, negative_slope, n_channels):
		super(CD_Autoencoder, self).__init__(device, n_clusters, input_dim, latent_dim, negative_slope)
		self.needsReshape = True
		self.n_channels = n_channels
		
		# Encoder
		self.encoder_model = nn.Sequential(
			nn.Conv2d(self.n_channels, 32, kernel_size = 5, stride = 2, padding = 2),
			nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
			nn.BatchNorm2d(32),

			nn.Conv2d(32, 64, kernel_size = 5, stride = 2, padding = 2),
			nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
			nn.BatchNorm2d(64),

			nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 0),
			nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
			nn.BatchNorm2d(128),

			nn.Flatten(start_dim=1),
			nn.Linear(128 * 3 * 3, self.latent_dim, bias=True), # latent_dim * 3 * 3
			nn.Tanh(),
			nn.BatchNorm1d(self.latent_dim), #TODO
		)
		
		# Clustering MLP - MLP Part from latent Dimension to Number of Clusters
		self.cluster_model = nn.Sequential(
	
			# Output Layer
			# nn.Linear(self.latent_dim, self.n_clusters, bias=True),
			rbf.RBF(self.latent_dim, self.n_clusters, self.kernel),
		)
		
		# Decoder 
		self.decoder_model = nn.Sequential(
			nn.Linear(self.latent_dim, 128 * 3 * 3, bias=True),
			nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
			nn.BatchNorm1d(128 * 3 * 3),
			nn.Unflatten(dim = 1, unflattened_size = (128, 3, 3)),

			nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 2, padding = 0),
			nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
			nn.BatchNorm2d(64),
			
			nn.ConvTranspose2d(64, 32, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
			nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
			nn.BatchNorm2d(32),
			
			nn.ConvTranspose2d(32, self.n_channels, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),            
			nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
			nn.BatchNorm2d(self.n_channels)
		)