import torch
import numbers
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from Evaluations.Evaluation import Evaluator
from torchvision import datasets, transforms
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))
    return dis_mat

class batch_KMeans(object):

    def __init__(self, latent_dim, n_clusters):
        self.n_features = latent_dim
        self.n_clusters = n_clusters
        self.clusters = np.zeros((self.n_clusters, self.n_features))
        self.count = 100 * np.ones((self.n_clusters)) 
        self.n_jobs = 1

    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_distance)(X, self.clusters[i])
            for i in range(self.n_clusters))
        dis_mat = np.hstack(dis_mat)

        return dis_mat

    def init_cluster(self, X, indices=None):
        """ Generate initial clusters using sklearn.Kmeans """
        model = KMeans(n_clusters=self.n_clusters,
                       n_init=20)
        model.fit(X)
        self.clusters = model.cluster_centers_  # copy clusters

    def update_cluster(self, X, cluster_idx):
        """ Update clusters in Kmeans on a batch of data """
        n_samples = X.shape[0]
        for i in range(n_samples):
            self.count[cluster_idx] += 1
            eta = 1.0 / self.count[cluster_idx]
            updated_cluster = ((1 - eta) * self.clusters[cluster_idx] +
                               eta * X[i])
            self.clusters[cluster_idx] = updated_cluster

    def update_assign(self, X):
        """ Assign samples in `X` to clusters """
        dis_mat = self._compute_dist(X)

        return np.argmin(dis_mat, axis=1)

class GenericDCNAutoencoder(nn.Module):

    def __init__(self, device, n_clusters, input_dim, latent_dim):
        super(GenericDCNAutoencoder, self).__init__()
        self.device = device
        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_channels = 1
        self.IMG_SIZE = 28

        self.kmeans = batch_KMeans(self.latent_dim, self.n_clusters)
        self.evaluator = Evaluator()

    def set_general_training_variables(self, dataloader, batch_size):
        self.dataloader = dataloader
        self.batch_size = batch_size

    def set_pretraining_variables(self, n_pret_epochs, pret_lr):
        self.n_pret_epochs = n_pret_epochs
        self.pret_lr = pret_lr

    def set_training_variables(self, n_epochs, lr, lamda, beta):
        self.n_epochs = n_epochs
        self.lr = lr
        self.lamda = lamda
        self.beta = beta

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
     
    def get_latent_data(self):
        data_list, latent_data_list, labels_list = list(), list(), list()

        for batch_index, (data, labels) in enumerate(self.dataloader):
            if(self.needsReshape):
                data = torch.reshape(data, (data.shape[0], 1, self.IMG_SIZE, self.IMG_SIZE)).to(self.device)
            else:    
                data = torch.reshape(data, (data.shape[0], (self.IMG_SIZE * self.IMG_SIZE))).to(self.device)

            data_list.append(data.cpu().detach().numpy())        

            code = self.encoder(data).to(self.device)
            code = code.cpu().detach().numpy()

            latent_data_list.append(code)
            labels_list.append(labels)

        return np.concatenate(data_list), np.concatenate(latent_data_list), np.concatenate(labels_list).astype(int)
    
    def get_cluster_centers(self):
        return self.kmeans.clusters

    def pretrain_autoencoder(self):
        MSE = nn.MSELoss().to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=self.pret_lr, weight_decay=1e-5)

        for epoch in range(self.n_pret_epochs):
            sum_rec_loss = 0
            for batch_index, (data, labels) in enumerate(self.dataloader):                             
                if(self.needsReshape):
                    data = torch.reshape(data, (data.shape[0], 1, self.IMG_SIZE, self.IMG_SIZE)).to(self.device)
                else:    
                    data = torch.reshape(data, (data.shape[0], (self.IMG_SIZE * self.IMG_SIZE))).to(self.device)

                rec_X = self.forward(data)
                loss = MSE(data, rec_X)
                """
                """
                reconstructions = self.forward(data)
                rec_loss = MSE(reconstructions, data)

                sum_rec_loss += rec_loss.item()

                optimizer.zero_grad()
                rec_loss.backward()
                optimizer.step()
                
            print(f'Epoch: {epoch}, Loss: {sum_rec_loss:.6f}')
        
        # Initialize clusters in self.kmeans after pre-training
        batch_X = []
        for batch_idx, (data, _) in enumerate(self.dataloader):
            if(self.needsReshape):
                data = torch.reshape(data, (data.shape[0], 1, self.IMG_SIZE, self.IMG_SIZE)).to(self.device)
            else:    
                data = torch.reshape(data, (data.shape[0], (self.IMG_SIZE * self.IMG_SIZE))).to(self.device)            
            latent_X = self.encoder(data)
            batch_X.append(latent_X.detach().cpu().numpy())
        batch_X = np.vstack(batch_X)
        self.kmeans.init_cluster(batch_X)

        return self
    
    def _loss(self, X, cluster_id):
        MSE = nn.MSELoss()

        rec_X = self.forward(X)
        latent_X = self.encoder(X)

        # Reconstruction error
        rec_loss = self.lamda * MSE(X, rec_X)

        # Regularization term on clustering
        dist_loss = torch.tensor(0.).to(self.device)
        clusters = torch.FloatTensor(self.kmeans.clusters).to(self.device)
        for i in range(X.size()[0]):
            diff_vec = latent_X[i] - clusters[cluster_id[i]]
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                            diff_vec.view(-1, 1))
            dist_loss += 0.5 * self.beta * torch.squeeze(sample_dist_loss)

        return (rec_loss + dist_loss,
                rec_loss.detach().cpu().numpy(),
                dist_loss.detach().cpu().numpy())

    def train_autoencoder(self):
        self.df_eval = pd.DataFrame(columns=['Loss','Rec_Loss','Dist_Loss','Accuracy','Purity','Nmi','Ari'])

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        MSE = nn.MSELoss()
        
        # Train Autoencoder
        for epoch in range(self.n_epochs):
            sum_loss = 0
            sum_rec_loss = 0
            sum_dist_loss = 0

            self.clusters_list = list()
            self.labels_list = list()
            self.latent_data_list = list()
            
            for batch_index, (data, labels) in enumerate(self.dataloader):
                
                if(self.needsReshape):
                    data = torch.reshape(data, (data.shape[0], 1, self.IMG_SIZE, self.IMG_SIZE)).to(self.device)
                else:    
                    data = torch.reshape(data, (data.shape[0], (self.IMG_SIZE * self.IMG_SIZE))).to(self.device)

                # Get the latent features
                with torch.no_grad():
                    """
                    latent_X = self.autoencoder(data, latent=True)
                    latent_X = latent_X.cpu().numpy()
                    """
                    code = self.encoder(data)
                    code = code.cpu().numpy()
                
                # [Step-1] Update the assignment results
                cluster_id = self.kmeans.update_assign(code)

                # [Step-2] Update clusters in bath Kmeans
                elem_count = np.bincount(cluster_id,
                                     minlength=self.n_clusters)
                for k in range(self.n_clusters):
                    # avoid empty slicing
                    if elem_count[k] == 0:
                        continue
                    self.kmeans.update_cluster(code[cluster_id == k], k)

                # [Step-3] Update the network parameters
                loss, rec_loss, dist_loss = self._loss(data, cluster_id)
                
                sum_loss += loss.item()
                sum_rec_loss += rec_loss.item()
                sum_dist_loss += dist_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.clusters_list.append(self.kmeans.update_assign(code))
                self.labels_list.append(labels)
                self.latent_data_list.append(code)
            
            self.clusters_list = np.concatenate(self.clusters_list).astype(int)
            self.labels_list = np.concatenate(self.labels_list).astype(int)
            self.latent_data_list = np.concatenate(self.latent_data_list)

            acc, pur, nmi, ari, sil = self.evaluator.evaluate_model(self.latent_data_list, self.labels_list, self.clusters_list)
            self.df_eval.loc[epoch] = [sum_loss, sum_rec_loss, sum_dist_loss, acc, pur, nmi, ari]
            print(f'Ep: {epoch} L: {sum_loss:.4f} Rec L: {sum_rec_loss:.4f} Dist L: {sum_dist_loss:.4f} ACC: {acc:.2f} PUR: {pur:.2f} NMI: {nmi:.2f} ARI: {ari:.2f}')

        return self.latent_data_list, self.labels_list, self.clusters_list
    
    def set_path(self):
        self.properties_name = str(self.n_epochs) + '_Eps'
        self.properties_name += '_ld_' + str(self.latent_dim) + '_out_' + str(self.n_clusters)
        self.properties_name += '_bs_' + str(self.batch_size) + '_lr_' + str(self.lr)
        self.properties_name += '_lamda_' + str(self.lamda) + '_beta_' + str(self.beta)
        self.data_dir_path = self.path_to_module + 'Results/' + self.dataset_name + '/DCNAE/' + self.properties_name

class DCNAutoencoder(GenericDCNAutoencoder):
    def __init__(self,  device, n_clusters, input_dim, latent_dim):
        super(DCNAutoencoder, self).__init__( device, n_clusters, input_dim, latent_dim)
        self.needsReshape = False
                
        self.encoder_model = nn.Sequential(
            nn.Linear(self.input_dim, 500),
            nn.ReLU(),

            nn.Linear(500, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),

            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.BatchNorm1d(2000),

            nn.Linear(2000, self.latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.latent_dim),
        )
    
        # Decoder Model - ([Latent Space, Linear], [2000, LeakyReLU], [500, LeakyReLU], [500, LeakyReLU], [Input Space, Linear])
        self.decoder_model = nn.Sequential(
          
            nn.Linear(self.latent_dim, 2000),
            nn.ReLU(),
            nn.BatchNorm1d(2000),

            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
    
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
    
            nn.Linear(500, self.input_dim)
        ) 

class DCNCDAutoencoder(GenericDCNAutoencoder):           
    def __init__(self, device, n_clusters, input_dim, latent_dim, n_channels):
        super(DCNCDAutoencoder, self).__init__(device, n_clusters, input_dim, latent_dim)
        self.needsReshape = True
        self.n_channels = n_channels
        
        # Encoder
        self.encoder_model = nn.Sequential(
            nn.Conv2d(self.n_channels, 32, kernel_size = 5, stride = 2, padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
    
            nn.Conv2d(32, 64, kernel_size = 5, stride = 2, padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
    
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
    
            nn.Flatten(start_dim=1),
            nn.Linear(128 * 3 * 3, self.latent_dim, bias=True), # latent_dim * 3 * 3
            nn.Tanh(),
            nn.BatchNorm1d(self.latent_dim), #TODO
        )
        
        # Decoder 
        self.decoder_model = nn.Sequential(
            nn.Linear(self.latent_dim, 128 * 3 * 3, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(128 * 3 * 3),
            nn.Unflatten(dim = 1, unflattened_size = (128, 3, 3)),
    
            nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 2, padding = 0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, self.n_channels, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),            
            nn.ReLU(),
            nn.BatchNorm2d(self.n_channels)
        )             