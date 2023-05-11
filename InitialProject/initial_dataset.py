import h5py
import pandas
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split


class InitialDataset(Dataset):
    def __init__(self,
                 path="../data/initial/train.h5",
                 variables_path="../data/initial/top_class_features.txt",
                 sorting_conditions=None,
                 label_name='Truth',
                 normalize_labels=False,):
        all_data = InitialDataset._load_data(path)
        with open(variables_path, 'r') as file:
            self.variables = file.read()
        self.variables = self.variables.replace("'", "")
        self.variables = self.variables.replace(" ", "")
        self.variables = self.variables.replace("\n", "")
        self.variables = self.variables.split(",")
        self.features = all_data[self.variables]
        self.labels = all_data[label_name]
        if isinstance(sorting_conditions, dict):
            for key, value in sorting_conditions.items():
                self.features = self.features[all_data[key] == value]

        # Convert to numpy and normalize
        self.labels = np.array(self.labels)
        self.features = np.array(self.features).T
        self.features = ((self.features - np.median(self.features))
                         / (np.quantile(self.features, 0.75) - np.quantile(self.features, 0.25)))
        self.features = self.features.T
        if normalize_labels:
            self.labels = (self.labels - np.mean(self.labels)) / np.std(self.labels)

        # Convert to tensor
        self.labels = torch.from_numpy(self.labels.reshape(-1, 1))
        self.features = torch.from_numpy(self.features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.features[item, :], self.labels[item]

    @staticmethod
    def _load_data(name):
        file_extension_length = len(".h5")
        with h5py.File(f'{name}', 'r') as f:
            filename = name.split('/')[-1][:-file_extension_length]
            return pandas.DataFrame(f[filename][:], dtype=np.float32)

    def split_data(self, train_fraction=0.8, seed=None):
        train_size = int(len(self) * train_fraction)
        if seed is None:
            return random_split(self, [train_size, len(self) - train_size])
        else:
            torch.manual_seed(seed)
            return random_split(self, [train_size, len(self) - train_size])


def main():
    data = InitialDataset()
    print(data.features)
    print(data.labels)
    print(data[42])

    data1 = InitialDataset(label_name='p_truth_E', sorting_conditions={'Truth': 1}, normalize_labels=True)
    print(len(data1))
    print(data1[42])

    train, test = data.split_data()
    print(len(train), len(test))


if __name__ == "__main__":
    main()
