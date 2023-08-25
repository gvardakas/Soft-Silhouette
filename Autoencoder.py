import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torch.optim as optim
import numpy as np
import pandas as pd
from New_Code_Files.SoftSilhouette import SoftSilhouette
from sklearn.cluster import KMeans
from New_Code_Files.Visualization import Visualization
from New_Code_Files.Evaluation import Evaluation

class GenericAutoencoder(nn.Module):
    def __init__(self, pathToModule, datasetName, device, preTrEpochs, dataloader, batchSize, preTrLR, lamda, trEpochs, trLR, latentDim, nClusters, negativeSlope, inputDim, kmeans_n_init):
        super(GenericAutoencoder, self).__init__()
        self.IMG_SIZE = 28
        self.datasetName = datasetName
        self.device = device
        self.preTrEpochs = preTrEpochs
        self.dataloader = dataloader
        self.batchSize = batchSize
        self.preTrLR = preTrLR
        self.lamda = lamda
        self.trEpochs = trEpochs
        self.trLR = trLR
        self.kmeans_n_init = kmeans_n_init
        self.evaluation = Evaluation(kmeans_n_init,'k-means++')
        self.softSilhouette = SoftSilhouette(self.device)
        self.nClusters = nClusters
        self.negativeSlope = negativeSlope
        self.inputDim = inputDim
        self.latentDim = latentDim
        self.pathToModule = pathToModule
        
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
    def forward_softMax(self, x):
        x = x.to(self.device)
        x = self.encoder_model(x)
        x = self.cluster_model(x)
        x = softmax(x, dim=1)
        return x
    
    def kmeans_initialization(self, n_init=10):
        self.kmeans = KMeans(n_clusters=self.nClusters, n_init=n_init).fit(self.preTrainedLatentdata)
        for weights in self.cluster_model[0].parameters():
            for index, center in enumerate(self.kmeans.cluster_centers_):
                with torch.no_grad():
                    center = torch.from_numpy(center) 
                    center.requires_grad_()
                    weights.data[index] = center
   
    def take_clusters(self):
        for weights in self.cluster_model[0].parameters():
            #for index in range(self.nClusters):
                #print(weights.data[index]) 
            #print(weights)
            return weights
            
    def pretrain_autoencoder(self):
        criterion = nn.MSELoss().to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=self.preTrLR)
        self.loss_list = list()

        self.train().to(self.device)
        for epoch in range(self.preTrEpochs):
            loss = 0
            self.preTrainedLatentdata = list()
            for batch_index, (batch, labels) in enumerate(self.dataloader):
                self.zero_grad()
                batch = batch.to(self.device)
                if self.needsReshape:
                    batch = torch.reshape(batch, (batch.shape[0], self.numOfChannels, self.IMG_SIZE, self.IMG_SIZE))
                
                reconstructions = self.forward(batch)
                train_loss = criterion(reconstructions, batch)
                
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
                
                # Take latent_data
                self.eval()
                code = self.encoder(batch).to(self.device)
                self.preTrainedLatentdata.append(code.cpu().detach().numpy())
                self.train()
                
            self.loss_list.append(loss)
            print("Epoch: {}/{}, Loss: {:.6f}".format(epoch + 1, self.preTrEpochs, loss))
        self.preTrainedLatentdata = np.concatenate(self.preTrainedLatentdata)
        self.eval()
        return        
    
    def save_pretrained_weights(self):
        
        # Set the file path where you want to save the model's state dictionary
        model_save_path = self.dataDirPath + "/Weigths/autoencoder_weights.pth"
        
        Visualization([],0,self).create_directory_if_not_exists(self.dataDirPath + "/Weigths")
        
        # Save the model's state dictionary
        torch.save(self.state_dict(), model_save_path)
        
    def train_autoencoder(self,silhouette_method="fast"):
        optimizer = optim.Adam(self.parameters(), lr=self.trLR)
        mse = nn.MSELoss()
        self.df = pd.DataFrame(columns=['Rec_Loss','Cl_Loss','Soft_Sil','Accuracy','Purity','Nmi','Ari'])
        
        # Train Autoencoder
        for epoch in range(self.trEpochs):
            self.clusters = list()
            self.realLabels = list()
            self.latentData = list()
            
            sumRecLoss = 0
            sumSoftSilhouette = 0
            sumClusteringLoss = 0
            
            #if(epoch!=0):
              #lamda = balanced_lamda_program(total_rec_loss,total_soft_sil)
            #print(lamda)
            #
            for batch_index, (realData, labels) in enumerate(self.dataloader):
                realData = realData.to(self.device)
                
                if self.needsReshape:
                    realData =  torch.reshape(realData, (realData.shape[0], self.numOfChannels, self.IMG_SIZE, self.IMG_SIZE))
                
                reconstruction = self.forward(realData)
                softClustering = self.forward_softMax(realData).to(self.device)
                
                
                ###################
                """
                max_conf, _ = torch.max(softClustering,axis=1)
                conf_indexes = torch.where(max_conf>0.65)[0]
                #print(conf_indexes)
                softClustering = softClustering[conf_indexes]
                realData = realData[conf_indexes]
                reconstruction = reconstruction[conf_indexes]
                labels = labels[conf_indexes.cpu().detach().numpy()]
                """
                ###################
                code = self.encoder(realData)
                
                # Take argmax from softClustering
                self.clusters.append(self.evaluation.clusters_to_numpy(softClustering))
                self.realLabels.append(labels.cpu().data.numpy())
                self.latentData.append(code.cpu().detach().numpy())
                
                if silhouette_method == "default":
                    softSil = self.softSilhouette.soft_silhouette(code, softClustering, requires_distance_grad=True)
                elif silhouette_method == "fast":
                    softSil = self.softSilhouette.fast_soft_silhouette(code, self.take_clusters() , softClustering, requires_distance_grad=True) 
                    
                rec = mse(reconstruction, realData)
                total_loss = rec + self.lamda * (1 - softSil)
        
                sumRecLoss += rec.item()
                sumClusteringLoss += self.lamda * (1 - softSil.item())
                sumSoftSilhouette += softSil.item()
        
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
           
            
            self.clusters = np.concatenate(self.clusters)
            self.realLabels = np.concatenate(self.realLabels)
            self.latentData = np.concatenate(self.latentData)
            acc, pur, nmi, ari = self.evaluation.autoencoder_evaluation(self.latentData,self.realLabels.astype(int),self.clusters)
            print(f"Epoch: {epoch} REC_LOSS: {sumRecLoss:.4f} CL_LOSS: {sumClusteringLoss:.4f} SIL: {sumSoftSilhouette:.4f} ACC: {acc:.2f} PUR: {pur:.2f} NMI: {nmi:.2f} ARI: {ari:.2f}")
            self.df.loc[epoch] = [sumRecLoss,sumClusteringLoss,sumSoftSilhouette,acc,pur,nmi,ari]
            
    def setPath(self):
        self.desPath =  self.datasetName+"_With_"+str(self.trEpochs)+'_Eps'
        if(self.h1AeActive):
          self.desPath += "_h1_" + str(self.h1AeDim) + "-" + str(self.h1AeActFunc)
        if(self.h2AeActive):
          self.desPath += "_h2_" + str(self.h2AeDim) + "-" + str(self.h2AeActFunc)
        if(self.h3AeActive):
          self.desPath += "_h3_" + str(self.h3AeDim) + "-" + str(self.h3AeActFunc)
        if(self.lsAeActive):
          self.desPath += "_ls_" + str(self.latentDim) + "-" + str(self.lsAeActFunc)
        self.desPath += "_out_" + str(self.inputDim) + "-" + str(self.outAeActFunc)
        if(self.h1MActive):
          self.desPath += "_h1m_" + str(self.h1MDim) + "-" + str(self.h1MActFunc)  
        if(self.h2MActive):
          self.desPath += "_h2m_" + str(self.h2MDim) + "-" + str(self.h2MActFunc)
        self.desPath += "_outm_"+str(self.nClusters) + "-" + str(self.outMActFunc)      
        self.desPath += "_pe_"+str(self.preTrEpochs)+"_pl_"+str(self.preTrLR)+"_bs_"+str(self.batchSize)+"_l_"+str(self.trLR)+"_lamda_"+str(self.lamda)
        self.dataDirPath = self.pathToModule + "/" + self.datasetName + "/AE_3_HL_LS/" + self.desPath
        
class Autoencoder(GenericAutoencoder):
    def __init__(self, pathToModule, datasetName, device, preTrEpochs, dataloader, batchSize, preTrLR, lamda, trEpochs, trLR, latentDim, nClusters, negativeSlope, inputDim, kmeans_n_init):
        super(Autoencoder, self).__init__(pathToModule, datasetName, device, preTrEpochs, dataloader, batchSize, preTrLR, lamda, trEpochs, trLR, latentDim, nClusters, negativeSlope, inputDim, kmeans_n_init)
        self.h1AeDim = 500
        self.h1AeActive = True
        self.h1AeActFunc = "LR"
        
        self.h2AeDim = 500
        self.h2AeActive = True
        self.h2AeActFunc = "LR"
        
        self.h3AeDim = 2000
        self.h3AeActive = True
        self.h3AeActFunc = "LR"
        
        self.lsAeActive = True
        self.lsAeActFunc = "TH"
        
        self.outAeActive = True
        self.outAeActFunc = "LR"
        
        self.h1MDim = 0
        self.h1MActFunc = ""
        self.h1MActive = False
        
        self.h2MDim = 0
        self.h2MActFunc = ""
        self.h2MActive = False
        
        self.outMActFunc = "L"
        
        self.needsReshape = False
        
        super().setPath()
        
        self.encoder_model = nn.Sequential(
            nn.Linear(self.inputDim, self.h1AeDim, bias = True),
            nn.LeakyReLU(negative_slope = self.negativeSlope, inplace=True),
            nn.BatchNorm1d(self.h1AeDim),

            nn.Linear(self.h1AeDim, self.h2AeDim, bias = True),
            nn.LeakyReLU(negative_slope = self.negativeSlope, inplace=True),
            nn.BatchNorm1d(self.h2AeDim),

            nn.Linear(self.h2AeDim, self.h3AeDim, bias = True),
            nn.LeakyReLU(negative_slope = self.negativeSlope, inplace=True),
            nn.BatchNorm1d(self.h3AeDim),

            nn.Linear(self.h3AeDim, self.latentDim, bias = True),
            nn.Tanh(),
            nn.BatchNorm1d(self.latentDim)
        )

        # Clustering MLP - MLP Part from latent Dimension to Number of Clusters
        self.cluster_model = nn.Sequential(
    
            # Output Layer
            nn.Linear(self.latentDim, self.nClusters, bias = False),
            #nn.LeakyReLU(negative_slope = self.negativeSlope, inplace=True),
            #nn.BatchNorm1d(self.nClusters) #WE HOLD IT
        )
    
        # Decoder Model - ([Latent Space, Linear], [2000, LeakyReLU], [500, LeakyReLU], [500, LeakyReLU], [Input Space, Linear])
        self.decoder_model = nn.Sequential(
            nn.Linear(self.latentDim, self.h3AeDim, bias = True),
            nn.LeakyReLU(negative_slope = self.negativeSlope, inplace=True),
            nn.BatchNorm1d(self.h3AeDim),
    
            nn.Linear(self.h3AeDim, self.h2AeDim, bias = True),
            nn.LeakyReLU(negative_slope = self.negativeSlope, inplace=True),
            nn.BatchNorm1d(self.h2AeDim),
    
            nn.Linear(self.h2AeDim, self.h1AeDim, bias = True),
            nn.LeakyReLU(negative_slope = self.negativeSlope, inplace=True),
            nn.BatchNorm1d(self.h1AeDim),
    
            nn.Linear(self.h1AeDim, self.inputDim, bias = True),
            nn.LeakyReLU(negative_slope = self.negativeSlope, inplace=True)
            #nn.BatchNorm1d(self.inputDim) APAGOREYETAI
        )
        
class CD_Autoencoder(GenericAutoencoder):
    def __init__(self, pathToModule, datasetName, device, preTrEpochs, dataloader, batchSize, preTrLR, lamda, trEpochs, trLR, latentDim, nClusters, negativeSlope, inputDim, kmeans_n_init, num_of_channels):
        super(CD_Autoencoder, self).__init__(pathToModule, datasetName, device, preTrEpochs, dataloader, batchSize, preTrLR, lamda, trEpochs, trLR, latentDim, nClusters, negativeSlope, inputDim, kmeans_n_init)
        self.h1AeDim = 32
        self.h1AeActive = True
        self.h1AeActFunc = "LR"
        
        self.h2AeDim = 64
        self.h2AeActive = True
        self.h2AeActFunc = "LR"
        
        self.h3AeDim = 128
        self.h3AeActive = True
        self.h3AeActFunc = "LR"
        
        self.lsAeActive = True
        self.lsAeActFunc = "TH"
        
        self.outAeActive = True
        self.outAeActFunc = "LR"
        
        self.h1MDim = 0
        self.h1MActFunc = ""
        self.h1MActive = False
        
        self.h2MDim = 0
        self.h2MActFunc = ""
        self.h2MActive = False
        
        self.outMActFunc = "L"
        
        self.needsReshape = True
        self.numOfChannels = num_of_channels
        super().setPath()
        
        # Encoder
        self.encoder_model = nn.Sequential(
            nn.Conv2d(self.numOfChannels, self.h1AeDim, kernel_size = 5, stride = 2, padding = 2),
            nn.LeakyReLU(negative_slope=self.negativeSlope, inplace=True),
            nn.BatchNorm2d(self.h1AeDim),

            nn.Conv2d(self.h1AeDim, self.h2AeDim, kernel_size = 5, stride = 2, padding = 2),
            nn.LeakyReLU(negative_slope=self.negativeSlope, inplace=True),
            nn.BatchNorm2d(self.h2AeDim),

            nn.Conv2d(self.h2AeDim, self.h3AeDim, kernel_size = 3, stride = 2, padding = 0),
            nn.LeakyReLU(negative_slope=self.negativeSlope, inplace=True),
            nn.BatchNorm2d(self.h3AeDim),

            nn.Flatten(start_dim=1),
            nn.Linear(self.h3AeDim * 3 * 3, self.latentDim, bias=True), # latentDim * 3 * 3
            nn.Tanh(),
            nn.BatchNorm1d(self.latentDim), #TODO
        )
        
        # Clustering MLP - MLP Part from latent Dimension to Number of Clusters
        self.cluster_model = nn.Sequential(
    
            # Output Layer
            nn.Linear(self.latentDim, self.nClusters, bias = False),
            #nn.LeakyReLU(negative_slope = self.negativeSlope, inplace=True),
            #nn.BatchNorm1d(self.nClusters) #WE HOLD IT
        )
        
        # Decoder 
        self.decoder_model = nn.Sequential(
            nn.Linear(self.latentDim,  self.h3AeDim * 3 * 3, bias=True),
            nn.LeakyReLU(negative_slope=self.negativeSlope, inplace=True),
            nn.BatchNorm1d(self.h3AeDim * 3 * 3),
            nn.Unflatten(dim = 1, unflattened_size = (self.h3AeDim, 3, 3)),

            nn.ConvTranspose2d(self.h3AeDim, self.h2AeDim, kernel_size = 3, stride = 2, padding = 0),
            nn.LeakyReLU(negative_slope=self.negativeSlope, inplace=True),
            nn.BatchNorm2d(self.h2AeDim),
            
            nn.ConvTranspose2d(self.h2AeDim, self.h1AeDim, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.LeakyReLU(negative_slope=self.negativeSlope, inplace=True),
            nn.BatchNorm2d(self.h1AeDim),
            
            nn.ConvTranspose2d(self.h1AeDim, self.numOfChannels, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),            
            nn.LeakyReLU(negative_slope=self.negativeSlope, inplace=True),
            nn.BatchNorm2d(self.numOfChannels)
        )
    
def createAutoencoder(pathToModule, datasetName, inputDim, negativeSlope, nClusters, device, preTrEpochs, dataloader, batchSize, preTrLR, lamda, trEpochs, trLR, latentDim, isCD, kmeans_n_init, num_of_channels):
    if isCD:
        # Create an instance of your autoencoder
        autoencoder = CD_Autoencoder(pathToModule, datasetName, device, preTrEpochs, dataloader, batchSize, preTrLR, lamda, trEpochs, trLR, latentDim, nClusters, negativeSlope, inputDim, kmeans_n_init, num_of_channels)
    else:    
        # Create an instance of your autoencoder
        autoencoder = Autoencoder(pathToModule, datasetName, device, preTrEpochs, dataloader, batchSize, preTrLR, lamda, trEpochs, trLR, latentDim, nClusters, negativeSlope, inputDim, kmeans_n_init)
    
    return autoencoder.to(device)


    

        
        
        
        