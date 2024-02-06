import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

SHUFFLE = True 

folder_path = './Datasets/'

def get_coil20(transform):
    dataset = datasets.ImageFolder(root=folder_path+'COIL-20/coil-20-proc', transform=transform)
    return dataset     

def get_cifar10(transform):
    return torchvision.datasets.CIFAR10(root=folder_path+'CIFAR-10/data', train=True, download=True, transform=transform)        

def get_stl10(transform):
    return torchvision.datasets.STL10(root=folder_path+'STL-10/data', split='train', download=True, transform=transform)     

def get_dataset(dataset_name, batch_size=64):
    function_name = "get_" + dataset_name
    function_to_call = globals()[function_name]
    
    IMG_SIZE = 28
    # Define the data transformations
    transform = transforms.Compose(
        [
         transforms.Resize((IMG_SIZE, IMG_SIZE)),
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor()])

    dataset = function_to_call(transform)
    
    # You can choose a subset of the dataset if needed
    subset_size = 10000  # Change this to the number of samples you want
    train_subset = Subset(dataset, range(subset_size))
    
    dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=SHUFFLE)

    return dataloader, 1, [], []
