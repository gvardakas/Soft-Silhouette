# Deep Clustering Using the Soft Silhouette Score: Towards Compact and Well-Separated Clusters

Unsupervised learning has gained prominence in the big data era, offering a means to extract valuable insights from unlabeled datasets. Deep clustering has emerged as an important unsupervised category, aiming to exploit the non-linear mapping capabilities of neural networks in order to enhance clustering performance. The majority of deep clustering literature focuses on minimizing the inner-cluster variability in some embedded space while keeping the learned representation consistent with the original high-dimensional dataset. In this work, we propose \emph{soft silhoutte}, a probabilistic formulation of the silhouette coefficient. Soft silhouette rewards compact and distinctly separated clustering solutions like the conventional silhouette coefficient. When optimized within a deep clustering framework, soft silhouette guides the learned representations towards forming compact and well-separated clusters. In addition, we introduce an autoencoder-based deep learning architecture that is suitable for optimizing the soft silhouette objective function. The proposed deep clustering method has been tested and compared with several well-studied deep clustering methods on various benchmark datasets, yielding very satisfactory clustering results.

# Execution Instructions

The project can be executed either using Docker (recommended) or by installing the required Python dependencies locally.

## Option 1: Docker (Recommended)

Docker provides a reproducible environment with all required dependencies, including Python, PyTorch, and JupyterLab.

The provided Docker configuration uses the CPU version of PyTorch and therefore does not require an NVIDIA GPU or a CUDA installation.

### Requirements

Before running the project, install:

- Docker Desktop
- Git

### Clone the Repository

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/gvardakas/Soft-Silhouette.git
cd Soft-Silhouette
```


### Build the Docker Image

From the root directory of the repository, run:

```bash
docker build -t soft-silhouette .
```

This command creates a Docker image containing the Python environment and all dependencies required to run the project.


### Run the Docker Container

On Windows PowerShell, run:

```powershell
docker run --name soft-silhouette-container -p 8888:8888 -v "${PWD}:/app" soft-silhouette
```

The repository directory is mounted to `/app` inside the Docker container. Therefore, changes made to the project files on the host system are immediately available inside the container.

After the container starts, JupyterLab will display a URL in the terminal similar to:

```text
http://127.0.0.1:8888/lab?token=...
```

Open this URL in your web browser.

In JupyterLab, open:

```text
DCSS_Demo.ipynb
```

and run the notebook cells sequentially.

## Option 2: Local Installation

Alternatively, the project can be executed without Docker.

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

Then open:

```text
DCSS_Demo.ipynb
```

using Jupyter Notebook, JupyterLab, Visual Studio Code, or another compatible environment, and run the notebook cells sequentially.

The original `requirements.txt` includes the CUDA 12.1 versions of PyTorch and torchvision. Therefore, users without a compatible NVIDIA GPU may need to install the corresponding CPU versions of PyTorch instead.

# References
```
@article{vardakas2024deep,
  title={Deep Clustering Using the Soft Silhouette Score: Towards Compact and Well-Separated Clusters},
  author={Vardakas, Georgios and Papakostas, Ioannis and Likas, Aristidis},
  journal={arXiv preprint arXiv:2402.00608},
  year={2024}
}

```
