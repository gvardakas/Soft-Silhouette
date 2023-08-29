import numpy as np
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment
from typing import Optional
import pdb

class Evaluator:

	def __init__(self):
		pass

	def evaluate_model(self, data, labels_true, labels_pred):		
		self.nmi = normalized_mutual_info_score(labels_true, labels_pred)
		self.ari = adjusted_rand_score(labels_true, labels_pred)
		self.acc = self.__compute_cluster_accuracy(labels_true, labels_pred)[0]
		self.pur = self.__compute_purity(labels_true, labels_pred)
		self.sil = 0 # silhouette_score(data, labels_pred)
		return self.acc, self.pur, self.nmi, self.ari, self.sil

	def get_cluster_accuracy(self):
		return self.cluster_accuracy

	def get_purity(self):
		return self.purity

	def get_nmi(self):
		return self.nmi

	def get_ari(self):
		return self.ari

	def print_evaluation(self):
		print('ACC: {:.2f} PUR: {:.2f} NMI: {:.2f} ARI: {:.2f}'.format(self.cluster_accuracy, self.purity, self.nmi, self.ari))

	def __compute_cluster_accuracy(self, labels_true, labels_pred, cluster_number: Optional[int] = None):
		"""
		Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
		determine reassignments.

		:param labels_true: list of true cluster numbers, an integer array 0-indexed
		:param labels_pred: list of predicted cluster numbers, an integer array 0-indexed
		:param cluster_number: number of clusters, if None then calculated from input
		:return: reassignment dictionary, clustering accuracy
		"""
		if cluster_number is None:
			# assume labels are 0-indexed
			cluster_number = (max(labels_pred.max(), labels_true.max()) + 1)
		
	
		count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
		for i in range(labels_pred.size):
			count_matrix[labels_pred[i], labels_true[i]] += 1

		row_ind, col_ind = linear_assignment(count_matrix.max() - count_matrix)
		reassignment = dict(zip(row_ind, col_ind))
		accuracy = count_matrix[row_ind, col_ind].sum() / labels_pred.size

		return accuracy, reassignment

	def __compute_purity(self, labels_true, labels_pred):
		"""
		Calculate the purity, a measurement of quality for the clustering 
		results.
		
		Each cluster is assigned to the class which is most frequent in the 
		cluster.  Using these classes, the percent accuracy is then calculated.
		
		Returns:
		  A number between 0 and 1.  Poor clusterings have a purity close to 0 
		  while a perfect clustering has a purity of 1.

		"""

		# get the set of unique cluster ids
		clusters = set(labels_pred)

		# find out what class is most frequent in each cluster
		correct = 0
		for cluster in clusters:
			# get the indices of rows in this cluster
			indices = np.where(labels_pred == cluster)[0]

			cluster_labels = labels_true[indices]
			majority_label = np.argmax(np.bincount(cluster_labels))
			correct += np.sum(cluster_labels == majority_label)
		
		return float(correct) / len(labels_pred)



'''

def main():
	evaluator = Evaluator()
	labels = np.array([0, 1, 2, 3])
	clusters = np.array([0, 1, 2, 3])
	
	evaluator.evaluate_clustering(labels, clusters)
	evaluator.print_evaluation()

if __name__ == '__main__':
	main()

'''