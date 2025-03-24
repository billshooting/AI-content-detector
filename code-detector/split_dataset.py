import os
from torch.utils.data import random_split, Dataset
import json


class RawDataset(Dataset):
    def __init__(self):
        super(RawDataset, self).__init__()
        openai_codes_path = os.path.join(os.path.dirname(__file__), "./data/ai_codes_dataset.json")
        deepseek_codes_path = os.path.join(os.path.dirname(__file__), "./data/deepseek_codes_dataset.json")
        human_codes_path = os.path.join(os.path.dirname(__file__), "./data/human_codes_dataset.json")

        dataset = []
        with open(openai_codes_path, 'r', encoding='utf-8') as f:
            dataset = dataset + json.load(f)
        with open(deepseek_codes_path, 'r', encoding='utf-8') as f:
            dataset = dataset + json.load(f)
        with open(human_codes_path, 'r', encoding='utf-8') as f:
            dataset = dataset + json.load(f)
        assert len(dataset) == 12000

        self.dataset: list = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: int):
        return self.dataset[index]

dataset = RawDataset()
train_dataset, test_dataset = random_split(dataset, [10000, len(dataset) - 10000])

stored_train_data = [d for d in train_dataset]
stored_test_data = [d for d in test_dataset]

train_data_path  = os.path.join(os.path.dirname(__file__), "./data/train_dataset.json")
with open(train_data_path, 'w+', encoding='utf-8') as f:
    json.dump(stored_train_data, f, indent=2)
    f.flush()

test_data_path = os.path.join(os.path.dirname(__file__), "./data/test_dataset.json")
with open(test_data_path, 'w+', encoding='utf-8') as f:
    json.dump(stored_test_data, f, indent=2)
    f.flush()