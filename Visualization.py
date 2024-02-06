import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator
import matplotlib.colors as mcolors

from sklearn.manifold import TSNE

from openpyxl import load_workbook,Workbook
import os.path

from General_Functions import General_Functions

class Visualization:

    def __init__(self):
        self.color_list = self.init_color_list()

    def init_color_list(self):
        color_list = list(mcolors.CSS4_COLORS.keys()) + list(mcolors.XKCD_COLORS.keys())
        np.random.shuffle(color_list)
        color_list = ['deepskyblue', 'orange', 'hotpink', 'limegreen'] + color_list
        return color_list
    
    def plot_image(self, image, label):
        plt.imshow(image.squeeze(), cmap='gray')  
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.show()

    def plot_collage(self, images, num_x_images, num_y_images, image_size, data_dir_path):
    
        # Create a blank canvas for the collage
        collage_size = (image_size[0] * num_x_images, image_size[1] * num_y_images)

        fig, axes = plt.subplots(num_x_images, num_y_images, figsize=(image_size[0], image_size[1]), facecolor = 'black')

        for i in range(num_x_images):
            for j in range(num_y_images):
                index = i * num_y_images + j
                image = images[index].reshape(image_size[0], image_size[1])
                
                axes[i, j].imshow(image, cmap='gray')
                axes[i, j].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(data_dir_path + "/Experiments/Collage.png", facecolor = 'white')
        plt.show()

    
    def plot_tsne(self, data, y_true, y_predict, cluster_centers, data_dir_path):
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
            plt.scatter(x=x, y=y, c=c) #, edgecolors='black')

        # Plot cluster centers
        plt.scatter(tsne_embeddings[:n_clusters, 0], tsne_embeddings[:n_clusters, 1], c='red', marker='x', s=500, linewidths=3, label='Cluster Centers')

        # Remove x-axis numbering and label
        plt.xticks([])  # Pass an empty list to remove ticks

        # Remove y-axis numbering and label
        plt.yticks([])  # Pass an empty list to remove ticks
        
        plt.axis('off')
        
        plt.tight_layout()
        
        exp_dir_path, experiment_name = General_Functions().save_plot(data_dir_path, "TSNE")
        plt.savefig(exp_dir_path + "/" + experiment_name + "_TSNE.png")
        plt.show() 
    
    def plot(self, data, y_true, y_predict, cluster_centers, data_dir_path):
        n_clusters = cluster_centers.shape[0]
        unique_labels = np.unique(y_true).astype(int)

        plt.figure(figsize=(10, 10))
        for label_id in unique_labels:
            selected_indexes = np.where(y_true == label_id)[0]
            x = data[selected_indexes, 0]
            y = data[selected_indexes, 1]
            c = self.color_list[label_id]
            plt.scatter(x=x, y=y, c=[c] * selected_indexes.shape[0]) #, edgecolors='silver')

        # Plot cluster centers
        #plt.scatter(cluster_centers[:n_clusters, 0], cluster_centers[:n_clusters, 1], c='red', marker='x', s=500, linewidths=3, label='Cluster Centers')

        # Remove x-axis numbering and label
        plt.xticks([])  # Pass an empty list to remove ticks

        # Remove y-axis numbering and label
        plt.yticks([])  # Pass an empty list to remove ticks

        plt.tight_layout()
        
        plt.axis('off')
        
        exp_dir_path, experiment_name = General_Functions().save_plot(data_dir_path, "No_TSNE")
        plt.savefig(exp_dir_path + "/" + experiment_name + ".png")
        plt.show()   

    def plot_3D(self, data, y_true, y_predict, cluster_centers, data_dir_path):
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
        
        exp_dir_path, experiment_name = General_Functions().save_plot(data_dir_path, "3D")
        plt.savefig(exp_dir_path + "/" + experiment_name + "_3D.png")
        plt.show()    

