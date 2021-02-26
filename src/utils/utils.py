import torch
from torch.utils.data import Dataset, DataLoader

def generate_batches(dataset, batch_size, shuffle=True, drop_last=True):
    """
    A generator function which wraps the Pytorck DataLoader. 
    It will ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict