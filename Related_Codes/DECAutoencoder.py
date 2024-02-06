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
from General_Functions import General_Functions
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, input_dim, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(n_clusters, input_dim))
    
    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.centers) ** 2, dim=2)
        q = 1.0 / (1.0 + norm_squared / self.alpha)
        q = (q ** ((self.alpha + 1.0) / 2.0)).transpose(0, 1)
        q = q / q.sum(1, keepdim=True)
        return q

class GenericDECAutoencoder(nn.Module):

    def __init__(self, device, n_clusters, input_dim, latent_dim, negative_slope):
        super(GenericDECAutoencoder, self).__init__()
        self.device = device
        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.IMG_SIZE = 28
        self.negative_slope = negative_slope
        self.evaluator = Evaluator()

    def set_general_training_variables(self, dataloader, batch_size):
        self.dataloader = dataloader
        self.batch_size = batch_size

    def set_pretraining_variables(self, n_pret_epochs, pret_lr):
        self.n_pret_epochs = n_pret_epochs
        self.pret_lr = pret_lr

    def set_training_variables(self, n_epochs, lr, momentum, alpha):
        self.n_epochs = n_epochs
        self.lr = lr
        self.momentum = momentum
        self.alpha = alpha
        self.clusteringLayer = ClusteringLayer(self.n_clusters, self.latent_dim, self.alpha)

    def set_path_variables(self, path_to_module, dataset_name):
        self.path_to_module = path_to_module
        self.dataset_name = dataset_name    

    def forward(self, x):
        x = x.to(self.device)
        x = self.encoder_model(x)
        x = self.decoder_model(x)
        return x

    def forward_clustering(self, x):
        x = x.to(self.device)
        x = self.encoder_model(x)
        x = self.clusteringLayer(x)
        return x

    def encoder(self, x):
        x = x.to(self.device)
        x = self.encoder_model(x)
        return x
     
    def target_distribution(self, q):
        weight = (q ** 2) / q.sum(0)
        return (weight.t() / weight.sum(1)).t()
    
    def get_latent_data(self):
        data_list, latent_data_list, labels_list = list(), list(), list()

        for batch_index, (data, labels) in enumerate(self.dataloader):
            data = data.to(self.device)
            
            if(self.needsReshape):
                data = torch.reshape(data, (data.shape[0], 1, self.IMG_SIZE, self.IMG_SIZE)).to(self.device)
                
            data_list.append(data.cpu().detach().numpy())        

            code = self.encoder(data).to(self.device)
            code = code.cpu().detach().numpy()

            latent_data_list.append(code)
            labels_list.append(labels)

        return np.concatenate(data_list), np.concatenate(latent_data_list), np.concatenate(labels_list).astype(int)
    
    def kmeans_initialization(self, n_init=20):
        _, latent_data, labels = self.get_latent_data()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=n_init).fit(latent_data)
        init_centers = torch.from_numpy(kmeans.cluster_centers_).to(self.device)
        self.clusteringLayer.centers = nn.Parameter(init_centers)
        self.evaluator.evaluate_model(latent_data, labels, kmeans.labels_)
        self.evaluator.print_evaluation()

    def get_cluster_centers(self):
        return self.clusteringLayer.centers.cpu().detach().numpy()

    def save_pretrained_weights(self):
        model_save_path = self.data_dir_path + "/Weigths/autoencoder_weights.pth"
        General_Functions().create_directory(self.data_dir_path + "/Weigths")
        torch.save(self.state_dict(), model_save_path)
    
    def pretrain_autoencoder(self):
        MSE = nn.MSELoss().to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=self.pret_lr, weight_decay=1e-5)

        for epoch in range(self.n_pret_epochs):
            sum_rec_loss = 0
            for batch_index, (data, labels) in enumerate(self.dataloader):
                data = data.to(self.device)                             

                if(self.needsReshape):
                    data = torch.reshape(data, (data.shape[0], 1, self.IMG_SIZE, self.IMG_SIZE)).to(self.device)
               
                reconstructions = self.forward(data)
                rec_loss = MSE(reconstructions, data)

                sum_rec_loss += rec_loss.item()

                optimizer.zero_grad()
                rec_loss.backward()
                optimizer.step()
                
            print(f'Epoch: {epoch}, Loss: {sum_rec_loss:.6f}')
        return self

    def train_autoencoder(self):
        self.df_eval = pd.DataFrame(columns=['Rec_Loss','Cl_Loss','Accuracy','Purity','Nmi','Ari'])

        optimizer = optim.SGD(params=self.parameters(), lr=self.lr, momentum=self.momentum)
        MSE = nn.MSELoss()
        KLDivLoss = nn.KLDivLoss(size_average=False)
        
        # Train Autoencoder
        for epoch in range(self.n_epochs):
            sum_rec_loss = 0
            sum_loss = 0

            self.clusters_list = list()
            self.labels_list = list()
            self.latent_data_list = list()
            
            for batch_index, (data, labels) in enumerate(self.dataloader):
                data = data.to(self.device)

                if(self.needsReshape):
                    data = torch.reshape(data, (data.shape[0], 1, self.IMG_SIZE, self.IMG_SIZE)).to(self.device)
              
                code = self.encoder(data)
                reconstructions = self.forward(data)
                output = self.forward_clustering(data)
                target = self.target_distribution(output).detach()
                
                rec_loss = MSE(reconstructions, data)
                loss = KLDivLoss(output.log(), target) / output.shape[0]
                
                sum_rec_loss += rec_loss.item()
                sum_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                self.clusters_list.append(torch.argmax(output.transpose(0,1), dim=1).cpu().detach().numpy())
                self.labels_list.append(labels)
                self.latent_data_list.append(code.cpu().detach().numpy())
                
            self.clusters_list = np.concatenate(self.clusters_list).astype(int)
            self.labels_list = np.concatenate(self.labels_list).astype(int)
            self.latent_data_list = np.concatenate(self.latent_data_list)

            acc, pur, nmi, ari, _ = self.evaluator.evaluate_model(self.latent_data_list, self.labels_list, self.clusters_list)
            self.df_eval.loc[epoch] = [sum_rec_loss, sum_loss, acc, pur, nmi, ari]
            print(f'Ep: {epoch} Rec L: {sum_rec_loss:.4f} L: {sum_loss:.4f} ACC: {acc:.2f} PUR: {pur:.2f} NMI: {nmi:.2f} ARI: {ari:.2f}')

        return self.latent_data_list, self.labels_list, self.clusters_list
    
    def set_path(self):
        self.properties_name = str(self.n_epochs) + '_Eps'
        self.properties_name += '_ld_' + str(self.latent_dim) + '_out_' + str(self.n_clusters)
        self.properties_name += '_bs_' + str(self.batch_size) + '_lr_' + str(self.lr)
        self.properties_name += '_momentum_' + str(self.momentum) + '_alpha_' + str(self.alpha)
        self.data_dir_path = self.path_to_module + 'Results/' + self.dataset_name + '/DECAE/' + self.properties_name

class DECAutoencoder(GenericDECAutoencoder):
    def __init__(self,  device, n_clusters, input_dim, latent_dim, negative_slope):
        super(DECAutoencoder, self).__init__(device, n_clusters, input_dim, latent_dim, negative_slope)
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
            #nn.LeakyReLU(negative_slope = self.negative_slope, inplace=True),
            nn.Tanh(),
            nn.BatchNorm1d(self.latent_dim)
        )
    
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

class DECCDAutoencoder(GenericDECAutoencoder):           
    def __init__(self, device, n_clusters, input_dim, latent_dim, negative_slope):
        super(DECCDAutoencoder, self).__init__(device, n_clusters, input_dim, latent_dim, negative_slope)
        self.needsReshape = True

        self.encoder_model = nn.Sequential(
            nn.Conv2d(self.input_dim, 32, kernel_size = 5, stride = 2, padding = 2),
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
            
            nn.ConvTranspose2d(32, self.input_dim, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),            
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True)
        )             

