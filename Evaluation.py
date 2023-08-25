from sklearn.cluster import KMeans
from evaluations.evaluation import Evaluator
from sklearn.metrics import silhouette_score

import torch
class Evaluation:
    def __init__(self,n_init,init):
        self.n_init = n_init
        self.init = init
    def kmeans_evaluation(self,n_clusters, data, labels):
        kmeans_labels = KMeans(n_clusters = n_clusters, n_init = self.n_init, init = self.init).fit_predict(data)
    
        evaluator = Evaluator()
        evaluator.evaluate_clustering(labels, kmeans_labels)
        evaluator.print_evaluation()
    
        ACC = evaluator.get_cluster_accuracy()
        PUR = evaluator.get_purity()
        NMI = evaluator.get_nmi()
        ARI = evaluator.get_ari()
        SIL = silhouette_score(data, labels)
        return ACC, PUR, NMI, ARI, SIL

    def clusters_to_numpy(self,clusters):
        # Get the data clusters based on max neuron
        clusters = torch.argmax(clusters, dim=1)
        clusters = clusters.cpu().detach().numpy()
        return clusters
        
    def autoencoder_evaluation(self,data,labels, autoencoder_labels):
        evaluator = Evaluator()
        evaluator.evaluate_clustering(labels, autoencoder_labels)
    
        ACC = evaluator.get_cluster_accuracy()
        PUR = evaluator.get_purity()
        NMI = evaluator.get_nmi()
        ARI = evaluator.get_ari()
        SIL = silhouette_score(data, autoencoder_labels)
        return ACC, PUR, NMI, ARI, SIL      

