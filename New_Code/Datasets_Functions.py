import pickle
import importlib

def createHashMap(filePath):
    # Create or load the hashmap from a file
    filename = filePath+'datasets_hashmap.pkl'
    
    try:
        # Try to load the existing hashmap from the file
        with open(filename, 'rb') as file:
            hashmap = pickle.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create a new empty hashmap
        hashmap = {}
    
    # Add or modify key-value pairs
    #hashmap['10x73k'] = [256, 8, 0, False,'datasets.datasets_New','load_10x73k','']
    #hashmap['TCGA'] = [256, 5, 0, False,'datasets.datasets_New','load_TCGA','']
    #hashmap['coil20'] = [256, 20, 0, False,'datasets.datasets_New','load_coil20_dataloader','']
    #hashmap['MNIST_subset'] = [256, 10, 10000, True, 'datasets.datasets_New', 'load_MNIST_subset_dataloader', '']
    #hashmap['train_test_MNIST'] = [256, 10, 10000, True, 'datasets.datasets_New', 'load_train_test_MNIST_dataloader', '']
    #hashmap['MNIST_last35k'] = [256, 10, 0, False,'datasets.datasets_New','load_MNIST_last35k_dataloader','']
    #hashmap['eMNIST_balanced_letters'] = [256, 37, 0, False,'datasets.datasets_New','load_eMNIST_dataloader','balanced letters']
    #hashmap['eMNIST_mnist'] = [256, 10, 0, False,'datasets.datasets_New','load_eMNIST_dataloader','mnist']
    #hashmap['eMNIST_balanced_digits'] = [256, 10, 0, False,'datasets.datasets_New','load_eMNIST_dataloader','balanced digits']
    #hashmap['train_test_MNIST'] = [256, 10, 10000, True, 'datasets.datasets_New', 'load_train_test_MNIST_dataloader', '']
    #hashmap['fashionMNIST_subset'] = [256, 20, 0, False,'datasets.datasets_New','load_fashionMNIST_subset_dataloader','']
    #hashmap['Pendigits'] = [256, 20, 0, False,'datasets.datasets_New','load_Pendigits','']
    # Save the hashmap to the file
    with open(filename, 'wb') as file:
        pickle.dump(hashmap, file)
        
def getHashMap(filePath):
    # Load the hashmap from a file
    filename = filePath+'datasets_hashmap.pkl'

    try:
        with open(filename, 'rb') as file:
            hashmap = pickle.load(file)
            return hashmap
    except FileNotFoundError:
        print("Hashmap file not found.")      
        return {}
        
def appendInHashMap(has_key,key,value,filePath):
    import pickle

    # Load the hashmap from a file
    filename = filePath+'datasets_hashmap.pkl'
    
    try:
        with open(filename, 'rb') as file:
            hashmap = pickle.load(file)
    except FileNotFoundError:
        print("Hashmap file not found.")
        hashmap = {}
    
    # Append a value to an existing or not key
    hashmap[has_key][key] = value
    
    # Save the updated hashmap to the file
    with open(filename, 'wb') as file:
        pickle.dump(hashmap, file)
    
    print(hashmap)
    return hashmap
def deleteFromHashMap(key,filePath):
    import pickle

    # Load the hashmap from a file
    filename = filePath+'datasets_hashmap.pkl'
    
    try:
        with open(filename, 'rb') as file:
            hashmap = pickle.load(file)
    except FileNotFoundError:
        print("Hashmap file not found.")
        hashmap = {}
    
    # Append a value to an existing key
    if key in hashmap:
        hashmap.pop(key)
    
    # Save the updated hashmap to the file
    with open(filename, 'wb') as file:
        pickle.dump(hashmap, file)
    
    print(hashmap)
    return hashmap

def updateHashMap(key,value,filePath):
    import pickle

    # Load the hashmap from a file
    filename = filePath+'datasets_hashmap.pkl'
    
    try:
        with open(filename, 'rb') as file:
            hashmap = pickle.load(file)
    except FileNotFoundError:
        print("Hashmap file not found.")
        hashmap = {}
    
    hashmap[key] = value
    
    # Save the updated hashmap to the file
    with open(filename, 'wb') as file:
        pickle.dump(hashmap, file)
    
    print(hashmap)
    return hashmap

def makeHashMapWithListsToHashmapWithDicts(filePath,hash_map_with_lists, keys_list):
    hash_map_with_dicts = {}

    for key, value_list in hash_map_with_lists.items():
        # Check if the number of keys matches the number of values in the list
        if len(keys_list) != len(value_list):
            raise ValueError("Number of keys and values in the list do not match.")
        
        # Convert the list to a dictionary using the keys from the keys_list
        value_dict = {str(keys_list[i]): value_list[i] for i in range(len(value_list))}
        hash_map_with_dicts[key] = value_dict
    
    # Create or load the hashmap from a file
    filename = filePath+'datasets_hashmap_v2.pkl'

    # Save the hashmap to the file
    with open(filename, 'wb') as file:
        pickle.dump(hash_map_with_dicts, file)
    return hash_map_with_dicts
    

def functionGetDataset(datasetProperties):
    # Import the module dynamically
    module = importlib.import_module(datasetProperties['module_name'])
    # Get the function dynamically
    function = getattr(module, datasetProperties['function_name'])
    return function(batch_size=datasetProperties['batch_size'], option_name=datasetProperties['option_name'])
        
        
        
        
        
        
        