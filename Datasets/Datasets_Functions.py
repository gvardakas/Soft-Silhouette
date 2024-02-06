import pickle
import importlib

def create_hashmap(file_path):
    file_name = file_path + 'datasets_hashmap.pkl'
    
    try:
        with open(file_name, 'rb') as file:
            hashmap = pickle.load(file)
    except FileNotFoundError:
        hashmap = {}
    
    with open(file_name, 'wb') as file:
        pickle.dump(hashmap, file)
        
def get_hashmap(file_path):
    file_name = file_path + 'datasets_hashmap.pkl'

    try:
        with open(file_name, 'rb') as file:
            hashmap = pickle.load(file)
            return hashmap
    except FileNotFoundError:
        print("Hashmap file not found.")      
        return {}
        
def append_in_hashmap(has_key, key, value, file_path):
    file_name = file_path + 'datasets_hashmap.pkl'
    
    try:
        with open(file_name, 'rb') as file:
            hashmap = pickle.load(file)
    except FileNotFoundError:
        print("Hashmap file not found.")
        hashmap = {}
    
    hashmap[has_key][key] = value
    
    with open(file_name, 'wb') as file:
        pickle.dump(hashmap, file)
    
    return hashmap

def delete_from_hashmap(key, file_path):
    file_name = file_path + 'datasets_hashmap.pkl'
    
    try:
        with open(file_name, 'rb') as file:
            hashmap = pickle.load(file)
    except FileNotFoundError:
        print("Hashmap file not found.")
        hashmap = {}
    
    if key in hashmap:
        hashmap.pop(key)
    
    with open(file_name, 'wb') as file:
        pickle.dump(hashmap, file)
    
    return hashmap

def delete_from_inner_hashmap(keys, file_path):
    file_name = file_path + 'datasets_hashmap.pkl'
    
    try:
        with open(file_name, 'rb') as file:
            hashmap = pickle.load(file)
    except FileNotFoundError:
        print("Hashmap file not found.")
        hashmap = {}
    
    if keys[0] in hashmap:
        if keys[1] in hashmap[keys[0]]:
            hashmap[keys[0]].pop(keys[1])
    
    with open(file_name, 'wb') as file:
        pickle.dump(hashmap, file)
    
    return hashmap

def update_outer_hashmap(key, value, file_path):
    file_name = file_path + 'datasets_hashmap.pkl'
    
    try:
        with open(file_name, 'rb') as file:
            hashmap = pickle.load(file)
    except FileNotFoundError:
        print("Hashmap file not found.")
        hashmap = {}
    
    hashmap[key] = value
    
    with open(file_name, 'wb') as file:
        pickle.dump(hashmap, file)
        
    return hashmap

def update_inner_hashmap(keys, new_value, file_path):
    file_name = file_path + 'datasets_hashmap.pkl'
    
    try:
        with open(file_name, 'rb') as file:
            hashmap = pickle.load(file)
    except FileNotFoundError:
        print("Hashmap file not found.")
        hashmap = {}
    
    if keys[0] in hashmap:
        current_dict = hashmap[keys[0]]
        if keys[1] in current_dict:
            current_dict[keys[1]] = new_value
        else:
            raise KeyError(f"Key '{keys[1]}' not found in dictionary.")
    else:
        raise KeyError(f"Key '{keys[0]}' not found in dictionary.")
        
    with open(file_name, 'wb') as file:
        pickle.dump(hashmap, file)    

def function_get_dataset(dataset_name, module_name, batch_size, n_clusters):
    module = importlib.import_module(module_name)
    function = getattr(module, "get_dataset")
    
    return function(dataset_name, batch_size)
        
        
        
        
        
        
        
