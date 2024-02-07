import torch

class Objectives:

	def __init__(self, device):
		self.device = device

	def soft_silhouette(self, X, soft_clustering, requires_distance_grad=False):
		if not requires_distance_grad: X = X.detach()
	  
		distances = torch.cdist(X, X, p=2).to(self.device)
		alphas = torch.matmul(distances, soft_clustering).to(self.device)
		n_data, n_clusters = alphas.shape
		
		# Calculate betas without explicit loops using tensor operations
		alphas_without_j = alphas.unsqueeze(2).repeat(1, 1, n_clusters)
		mask = ~torch.eye(n_clusters, dtype=torch.bool).to(alphas.device).unsqueeze(0)
		alphas_without_self = torch.masked_select(alphas_without_j, mask).view(n_data, n_clusters - 1, n_clusters)
		betas, _ = torch.min(alphas_without_self, dim=1)
		# reversed_betas
		betas = betas[:, torch.arange(betas.size(1)-1, -1, -1)]
		
		#sc = torch.div(torch.sub(betas, alphas), torch.max(alphas, betas)).to(self.device)
		sc = torch.div(torch.sub(betas, alphas), torch.max(alphas, betas)).to(self.device)
		s = torch.mean(torch.sum(torch.mul(soft_clustering, sc), dim=1)).to(self.device)
		return s

	def entropy(self, probabilities, base=2):
		"""
		Compute the entropy of a probability distribution while retaining gradients.

		:param probabilities: A tensor of probabilities (e.g., output of a softmax function).
		:param base: The logarithmic base for entropy calculation (default is 2 for bits).
		:return: The entropy of the distribution.
		"""
		# Ensure probabilities sum to 1 along the specified dimension (usually dim=1 for a batch of distributions).
		assert torch.allclose(probabilities.sum(dim=-1), torch.ones_like(probabilities.sum(dim=-1))), "Probabilities do not sum to 1."

		# Apply logarithm with the specified base and compute the entropy.
		log_probabilities = torch.log(probabilities + 1e-10).to(self.device)  # Add a small epsilon to avoid log(0).
		entropy_matrix = -torch.sum(probabilities * log_probabilities, dim=-1).to(self.device) / torch.log(torch.tensor(base, dtype=probabilities.dtype)).to(self.device)
		mean_entropy = torch.mean(entropy_matrix).to(self.device)

		return mean_entropy, entropy_matrix

		
