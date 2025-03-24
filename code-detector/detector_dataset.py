from torch.utils.data import Dataset
import os
import json

class DetectorDataset(Dataset):
    def __init__(self, path):
        super(DetectorDataset, self).__init__()

        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            dataset = dataset + json.load(f)

        self.dataset: list = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: int):
        return self.dataset[index]