import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import os.path
from os import path
from openpyxl import load_workbook,Workbook
import pandas as pd
from matplotlib.ticker import FixedLocator

class Visualization:
    def __init__(self,color_list,fontsize,model):
        self.color_list = color_list
        self.fontsize = fontsize
        self.model = model
        
    def plot_tsne(self, figxsize=10, figysize=10, mlp=False):
        
        plt.figure(figsize=(figxsize, figysize))
        unique_labels = np.unique(self.model.labels_list).astype(int)
        
        # Cluster with TSNE
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
        
        cluster_centers = self.model.take_clusters().cpu().detach().numpy()
        if mlp:
            tsne_embeddings = tsne.fit_transform(np.concatenate((cluster_centers, self.model.data_list)))
        else:
            tsne_embeddings = tsne.fit_transform(np.concatenate((cluster_centers, self.model.latent_data_list)))
        
        for label_id in unique_labels:
            selected_indexes = np.where(self.model.labels_list == label_id)[0]
            x = tsne_embeddings[self.model.n_clusters:][selected_indexes, 0]
            y = tsne_embeddings[self.model.n_clusters:][selected_indexes, 1]
            c = [self.color_list[label_id]] * selected_indexes.shape[0]
            plt.scatter(x=x, y=y, c=c, edgecolors='black')
        
        # Plot cluster centers
        plt.scatter(tsne_embeddings[:self.model.n_clusters, 0], tsne_embeddings[:self.model.n_clusters, 1], c='red', marker='x', s=300, label='Cluster Centers')
         
        # Remove x-axis numbering and label
        plt.xticks([])  # Pass an empty list to remove ticks

        # Remove y-axis numbering and label
        plt.yticks([])  # Pass an empty list to remove ticks

        plt.tight_layout()
        #self.create_directory_if_not_exists(self.model.expDirPath+ "\\TSNE")
        #plt.savefig(self.model.expDirPath + "\\TSNE\\" + self.model.experimentName+"_TSNE.png")
        #plt.show()    
    
    def plot(self, figxsize=10, figysize=10, mlp=False):
        
        plt.figure(figsize=(figxsize, figysize))
        
        unique_labels = np.unique(self.model.clusters_list).astype(int)
        
        # Cluster with TSNE
        if mlp:
            for label_id in unique_labels:
                selected_indexes = np.where(self.model.clusters_list == label_id)[0]
                x = self.model.data_list[selected_indexes, 0]
                y = self.model.data_list[selected_indexes, 1]
                c = [self.color_list[label_id]] * selected_indexes.shape[0]
                plt.scatter(x=x, y=y, c=c, edgecolors='black')
        else:
            for label_id in unique_labels:
                selected_indexes = np.where(self.model.clusters_list == label_id)[0]
                x = self.model.latentData[selected_indexes, 0]
                y = self.model.latentData[selected_indexes, 1]
                c = [self.color_list[label_id]] * selected_indexes.shape[0]
                plt.scatter(x=x, y=y, c=c, edgecolors='black')
        if mlp:
            cluster_centers = self.model.get_clustering_layer_centers().cpu().detach().numpy()
            # Plot cluster centers
            plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=300, label='Cluster Centers')
        
        # Remove x-axis numbering and label
        plt.xticks([])  # Pass an empty list to remove ticks

        # Remove y-axis numbering and label
        plt.yticks([])  # Pass an empty list to remove ticks

        plt.tight_layout()
        
        #self.create_directory_if_not_exists(self.model.expDirPath+ "\\No_TSNE")
        #plt.savefig(self.model.expDirPath + "\\No_TSNE\\" + self.model.experimentName+"_No_TSNE.png")
        #plt.show()    
    
    def plot_3D(self, figxsize=10, figysize=10, mlp=False):
        
        fig = plt.figure(figsize=(figxsize, figysize))
        ax = fig.add_subplot(111, projection='3d')
        
        unique_labels = np.unique(self.model.realLabels).astype(int)
        
        # Cluster with TSNE
        
        for label_id in unique_labels:
            selected_indexes = np.where(self.model.realLabels == label_id)[0]
            x = self.model.latentData[selected_indexes, 0]
            y = self.model.latentData[selected_indexes, 1]
            z = self.model.latentData[selected_indexes, 2]
            c = [self.color_list[label_id]] * selected_indexes.shape[0]
            ax.scatter(x, y, z, c=c)
        
        # Set labels
        ax.set_xlabel('$x$', fontsize = self.fontsize)
        ax.set_ylabel('$y$', fontsize = self.fontsize)
        ax.set_zlabel('$z$', fontsize = self.fontsize)
        
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        ax.set_zlim(-10,10)
        
        plt.tight_layout()
        
        self.create_directory_if_not_exists(self.model.expDirPath+ "\\3D")
        plt.savefig(self.model.expDirPath + "\\3D\\" + self.model.experimentName+"_3D.png")
        #plt.show()    
            
    def saveExperimentResultsToExcel(self):
        self.model.dataPath = self.model.dataDirPath + "\\Data.xlsx"
        if(path.exists(self.model.dataPath)!=True):
            dataWb = Workbook()
            dataWb.save(self.model.dataPath)

        excelDataWΒ = load_workbook(self.model.dataPath)
        sheetNameSplittedArray = excelDataWΒ.sheetnames[-1].split("_")
        
        if len(sheetNameSplittedArray) == 1:
            std=excelDataWΒ.get_sheet_by_name(sheetNameSplittedArray[0])
            excelDataWΒ.remove_sheet(std)
            self.model.experimentName = "Experiment_1"
        elif sheetNameSplittedArray[1] != "":
            self.model.experimentName = "Experiment_" + str(int(sheetNameSplittedArray[1]) + 1)
        
        dataWriter = pd.ExcelWriter(self.model.dataPath, engine = 'openpyxl')
        dataWriter.book = excelDataWΒ

        self.model.df.to_excel(dataWriter, sheet_name=self.model.experimentName)
        dataWriter.save()
        dataWriter.close()
        
    def makeExcel(self):
        self.create_directory_if_not_exists(self.model.dataDirPath)
        self.model.expDirPath = self.model.dataDirPath + "\\" + "Experiment_Plots"
        self.create_directory_if_not_exists(self.model.expDirPath)
        
        self.saveExperimentResultsToExcel()
        
    def create_directory_if_not_exists(self,directory_path):
        
        """Create a directory if it doesn't already exist."""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        else:
            print(f"Directory '{directory_path}' already exists.")    
    
