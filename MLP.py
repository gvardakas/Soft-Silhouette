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

class MLP(nn.Module):
    def __init__(self, pathToModule, datasetName, device, dataloader, batchSize, trEpochs, trLR, nClusters, negativeSlope, inputDim, kmeans_n_init):
        super(MLP, self).__init__()
        self.IMG_SIZE = 28
        self.datasetName = datasetName
        self.device = device
        self.dataloader = dataloader
        self.batchSize = batchSize
        self.trEpochs = trEpochs
        self.trLR = trLR
        self.kmeans_n_init = kmeans_n_init
        self.evaluation = Evaluation(kmeans_n_init,'k-means++')
        self.softSilhouette = SoftSilhouette(self.device)
        self.nClusters = nClusters
        self.negativeSlope = negativeSlope
        self.inputDim = inputDim
        self.pathToModule = pathToModule
        
        self.h1MDim = 0
        self.h1MActFunc = ""
        self.h1MActive = False
        
        self.h2MDim = 0
        self.h2MActFunc = ""
        self.h2MActive = False
        
        self.outMActFunc = "L"
        
        self.model = nn.Sequential(
                
                nn.Linear(inputDim, nClusters, bias=False),

                #nn.LeakyReLU(negativeSlope, inplace=True),
                #nn.ReLU(inplace=True),
                #nn.Sigmoid(),

                #nn.BatchNorm1d(nClusters)
            )
        self.setPath()    

    
    # Here we implement the forward of the data through mlp
    def forward(self, x):
        x = x.to(self.device)
        x = self.model(x)
        return x


    # Forward with Soft Max for data x through MLP Model (1st Part) and SoftMax Function (2nd Part)
    def forward_softMax(self, x):
        x = x.to(self.device)
        x = self.model(x)
        x = softmax(x, dim=1)
        return x
    
    def takeRealData(self):
        self.realData = list()
        for batch_index, (realData, labels) in enumerate(self.dataloader):
            realData = realData.to(self.device)
            self.realData.append(realData.cpu().data.numpy())
        self.realData = np.concatenate(self.realData)   
    
    def kmeans_initialization(self, n_init=10):
        self.takeRealData()    
        self.kmeans = KMeans(n_clusters=self.nClusters, n_init=n_init).fit(self.realData)
        for weights in self.model[0].parameters():
            for index, center in enumerate(self.kmeans.cluster_centers_):
                with torch.no_grad():
                    center = torch.from_numpy(center) 
                    center.requires_grad_()
                    weights.data[index] = center
   
    def take_clusters(self):
        for weights in self.model[0].parameters():
            #for index in range(self.nClusters):
                #print(weights.data[index]) 
            #print(weights)
            return weights
    
    def train_mlp(self,silhouette_method="fast"):
        optimizer = optim.Adam(self.parameters(), lr=self.trLR)
        self.df = pd.DataFrame(columns=['Cl_Loss','Soft_Sil','Accuracy','Purity','Nmi','Ari'])
        
        # Train MLP
        for epoch in range(self.trEpochs):
            self.realData = list()
            self.clusters = list()
            self.realLabels = list()
            
            sumSoftSilhouette = 0
            sumClusteringLoss = 0
            
            for batch_index, (realData, labels) in enumerate(self.dataloader):
                realData = realData.to(self.device)
                softClustering = self.forward_softMax(realData).to(self.device)
                self.softClustering = softClustering
                # Take argmax from softClustering
                self.realData.append(realData.cpu().data.numpy())
                self.clusters.append(softClustering)
                self.realLabels.append(labels.cpu().data.numpy())

                softSil = self.softSilhouette.soft_silhouette(realData, softClustering, requires_distance_grad=True)
                
                total_loss = 1 - softSil
        
                sumClusteringLoss += 1 - softSil.item()
                
                sumSoftSilhouette += softSil.item()
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
           
            
            self.clusters = torch.cat(self.clusters, dim=0)
            self.clusters = self.evaluation.clusters_to_numpy(self.clusters)
            self.realLabels = np.concatenate(self.realLabels)
            self.realData = np.concatenate(self.realData)   
            acc, pur, nmi, ari, sil = self.evaluation.autoencoder_evaluation(self.realData,self.realLabels.astype(int),self.clusters)
            print(f"Epoch: {epoch} CL_LOSS: {sumClusteringLoss:.4f} SOFT_SIL: {sumSoftSilhouette:.4f} SIL: {sil:.4f} ACC: {acc:.2f} PUR: {pur:.2f} NMI: {nmi:.2f} ARI: {ari:.2f}")
            self.df.loc[epoch] = [sumClusteringLoss,sumSoftSilhouette,acc,pur,nmi,ari]
            
    def setPath(self):
        self.desPath =  self.datasetName+"_With_"+str(self.trEpochs)+'_Eps'
        self.desPath += "_in_" + str(self.inputDim)
        if(self.h1MActive):
          self.desPath += "_h1_" + str(self.h1MDim) + "-" + str(self.h1MActFunc)  
        if(self.h2MActive):
          self.desPath += "_h2_" + str(self.h2MDim) + "-" + str(self.h2MActFunc)
        self.desPath += "_out_"+str(self.nClusters) + "-" + str(self.outMActFunc)      
        self.desPath += "_bs_"+str(self.batchSize)+"_l_"+str(self.trLR)
        self.dataDirPath = self.pathToModule + "/" + self.datasetName + "/MLP/" + self.desPath
            
def createMLP(pathToModule, datasetName, device, dataloader, batchSize, trEpochs, trLR, nClusters, negativeSlope, inputDim, kmeans_n_init):
    # Create an instance of your autoencoder
    mlp = MLP(pathToModule, datasetName, device, dataloader, batchSize, trEpochs, trLR, nClusters, negativeSlope, inputDim, kmeans_n_init)
    return mlp.to(device)


    

        
        
        
        
