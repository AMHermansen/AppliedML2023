import torch
from torch.utils.data import Dataset, random_split
import h5py
import pandas as pd
import numpy as np


class ParticleDataset(Dataset):
    def __init__(self, path="/home/amh/Documents/Coding/GitHub/AppliedML2023/data/initial/train",
                 variables_path="/home/amh/Documents/coding/GitHub/AppliedML2023/data/initial/classification_variables.txt",
                 target="Truth"):
        if target == "ALL":
            with open("/home/amh/Documents/coding/GitHub/AppliedML2023/data/initial/variables.txt", "r") as f:
                target = f.read()
            target = target.replace("\n", "").replace("'", "").split(",")

        with h5py.File(f"{path}.h5", "r") as f:
            data = pd.DataFrame(f[path.split('/')[-1]][:], dtype=np.float32)

        with open(variables_path, "r") as f:
            self.variables = f.read()
        self.variables = self.variables.replace("\n", "").replace(" ", "").replace("'", "")
        self.variables = self.variables.split(",")

        self.features = data[self.variables]
        self.target = data[target]

        self.features = np.array(self.features)
        self.target = np.array(self.target)
        self.features = (self.features - np.mean(self.features, axis=0)) / np.std(self.features, axis=0, ddof=1)
        np.nan_to_num(self.features, copy=False, nan=0.0, posinf=10.0, neginf=10.0)

        self.features = torch.from_numpy(self.features)
        self.target = torch.Tensor(self.target)

    def __getitem__(self, item):
        return self.features[item, :], self.target[item]

    def __len__(self):
        return len(self.target)

    def split_data(self, train_fraction, seed=42):
        train_size = int(train_fraction * len(self))
        return random_split(self, [train_size, len(self) - train_size], generator=torch.Generator().manual_seed(seed))


def main():
    d = ParticleDataset()
    print(d[42])
    print(d.features.shape)


if __name__ == "__main__":
    main()
