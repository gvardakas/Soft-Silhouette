import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator
import matplotlib.colors as mcolors


from sklearn.manifold import TSNE

from openpyxl import load_workbook,Workbook
import os.path

class Visualization:

    def __init__(self):
        self.color_list = self.init_color_list()

    def init_color_list(self):
        color_list = list(mcolors.CSS4_COLORS.keys()) + list(mcolors.XKCD_COLORS.keys())
        np.random.shuffle(color_list)
        color_list = ['deepskyblue', 'gold', 'hotpink', 'limegreen'] + color_list
        return color_list

    def plot_tsne(self, data, y_true, y_predict, cluster_centers):
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
        tsne_embeddings = tsne.fit_transform(np.concatenate((cluster_centers, data)))

        n_clusters = cluster_centers.shape[0]
        unique_labels = np.unique(y_true).astype(int)

        plt.figure(figsize=(10, 10))
        for label_id in unique_labels:
            selected_indexes = np.where(y_true == label_id)[0]
            x = tsne_embeddings[n_clusters:][selected_indexes, 0]
            y = tsne_embeddings[n_clusters:][selected_indexes, 1]
            c = [self.color_list[label_id]] * selected_indexes.shape[0]
            plt.scatter(x=x, y=y, c=c, edgecolors='black')

        # Plot cluster centers
        plt.scatter(tsne_embeddings[:n_clusters, 0], tsne_embeddings[:n_clusters, 1], c='red', marker='x', s=500, edgecolors='black', label='Cluster Centers')

        # Remove x-axis numbering and label
        plt.xticks([])  # Pass an empty list to remove ticks

        # Remove y-axis numbering and label
        plt.yticks([])  # Pass an empty list to remove ticks

        plt.tight_layout()
        #self.create_directory_if_not_exists(self.model.expDirPath+ "\\TSNE")
        #plt.savefig(self.model.expDirPath + "\\TSNE\\" + self.model.experimentName+"_TSNE.png")
        #plt.show() 

    def plot(self, data, y_true, y_predict, cluster_centers):
        n_clusters = cluster_centers.shape[0]
        unique_labels = np.unique(y_true).astype(int)

        plt.figure(figsize=(10, 10))
        for label_id in unique_labels:
            selected_indexes = np.where(y_true == label_id)[0]
            x = data[selected_indexes, 0]
            y = data[selected_indexes, 1]
            c = self.color_list[label_id]
            plt.scatter(x=x, y=y, c=[c] * selected_indexes.shape[0], edgecolors='black')

        # Plot cluster centers
        plt.scatter(cluster_centers[:n_clusters, 0], cluster_centers[:n_clusters, 1], c='red', marker='x', s=500, edgecolors='black', label='Cluster Centers')

        # Remove x-axis numbering and label
        plt.xticks([])  # Pass an empty list to remove ticks

        # Remove y-axis numbering and label
        plt.yticks([])  # Pass an empty list to remove ticks

        plt.tight_layout()
        #self.create_directory_if_not_exists(self.model.expDirPath+ "\\TSNE")
        #plt.savefig(self.model.expDirPath + "\\TSNE\\" + self.model.experimentName+"_TSNE.png")
        #plt.show()   

    def plot_3D(self, data, y_true, y_predict, cluster_centers):
        unique_labels = np.unique(y_true).astype(int)
        
        # Cluster with TSNE
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for label_id in unique_labels:
            selected_indexes = np.where(y_true == label_id)[0]
            x = data[selected_indexes, 0]
            y = data[selected_indexes, 1]
            z = data[selected_indexes, 2]
            c = [self.color_list[label_id]] * selected_indexes.shape[0]
            ax.scatter(x, y, z, c=c)
        
        # Set labels
        ax.set_xlabel('$x$', fontsize = 10)
        ax.set_ylabel('$y$', fontsize = 10)
        ax.set_zlabel('$z$', fontsize = 10)
        
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        ax.set_zlim(-10,10)
        
        plt.tight_layout()
        
        # self.create_directory_if_not_exists(self.model.expDirPath+ "\\3D")
        #plt.savefig(self.model.expDirPath + "\\3D\\" + self.model.experimentName+"_3D.png")
        #plt.show()    

