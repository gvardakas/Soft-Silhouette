# Deep Clustering Using the Soft Silhouette Score: Towards Compact and Well-Separated Clusters

Unsupervised learning has gained prominence in the big data era, offering a means to extract valuable insights from unlabeled datasets. Deep clustering has emerged as an important unsupervised category, aiming to exploit the non-linear mapping capabilities of neural networks in order to enhance clustering performance. The majority of deep clustering literature focuses on minimizing the inner-cluster variability in some embedded space while keeping the learned representation consistent with the original high-dimensional dataset. In this work, we propose \emph{soft silhoutte}, a probabilistic formulation of the silhouette coefficient. Soft silhouette rewards compact and distinctly separated clustering solutions like the conventional silhouette coefficient. When optimized within a deep clustering framework, soft silhouette guides the learned representations towards forming compact and well-separated clusters. In addition, we introduce an autoencoder-based deep learning architecture that is suitable for optimizing the soft silhouette objective function. The proposed deep clustering method has been tested and compared with several well-studied deep clustering methods on various benchmark datasets, yielding very satisfactory clustering results.

# Execution Instructions
1) You should first install all rewuirements via running in your terminal the command pip install -r requirements.txt.
2) You should open the DCSS_Demo.ipynb and run each cell of it via a suitable program e.g. Visual Code  

# References
```
@article{vardakas2024deep,
  title={Deep Clustering Using the Soft Silhouette Score: Towards Compact and Well-Separated Clusters},
  author={Vardakas, Georgios and Papakostas, Ioannis and Likas, Aristidis},
  journal={arXiv preprint arXiv:2402.00608},
  year={2024}
}

```
