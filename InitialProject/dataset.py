from typing import Tuple, List

import torch
from torch.utils.data import Dataset, random_split, Subset
import h5py
import pandas as pd
import numpy as np


APP_ML_PATH = "/home/amh/Documents/Coding/GitHub/AppliedML2023"


class ParticleDataset(Dataset):
    def __init__(self,
                 path=f"{APP_ML_PATH}/data/initial/train",
                 variables_path=f"{APP_ML_PATH}/data/initial/classification_variables.txt",
                 target="Truth"):
        if target == "ALL":
            with open(f"{APP_ML_PATH}/data/initial/variables.txt", "r") as f:
                self.target_variables = f.read()
            self.target_variables = self.target_variables.replace("\n", "").replace("'", "").replace(" ", "").split(",")
        else:
            self.target_variables = target

        with h5py.File(f"{path}.h5", "r") as f:
            data = pd.DataFrame(f[path.split('/')[-1]][:], dtype=np.float32)

        with open(variables_path, "r") as f:
            self.variables = f.read()

        self.variables = self.variables.replace("\n", "").replace(" ", "").replace("'", "")
        self.variables = self.variables.split(",")

        self.features = data[self.variables]
        self.target = data[self.target_variables]

        self.features = np.array(self.features)
        self.target = np.array(self.target)
        self.features = self._normalize(self.features)
        if target == "ALL":
            self.target = self._normalize(self.target)

        self.features = torch.from_numpy(self.features)
        self.target = torch.Tensor(self.target)
        self._use_2d_getitem_dispatcher = isinstance(self.target_variables, list)
        self._all_data = data

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._use_2d_getitem_dispatcher:
            return self._get_item2d(item)
        else:
            return self._get_item1d(item)

    def __len__(self) -> int:
        return len(self.target)

    def split_data(self, train_fraction, seed=42) -> List[Subset[Dataset]]:
        train_size = int(train_fraction * len(self))
        return random_split(self, [train_size, len(self) - train_size],
                            generator=torch.Generator().manual_seed(seed))

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        out = (features - np.mean(features, axis=0)) / (2 * np.std(features, axis=0, ddof=1))
        np.nan_to_num(out, copy=False, nan=0.0, posinf=10.0, neginf=10.0)
        return out

    def _get_item1d(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[item, :], self.target[item]

    def _get_item2d(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[item, :], self.target[item, :]


class AEDataset(ParticleDataset):
    def __init__(
            self,
            path=f"{APP_ML_PATH}/data/initial/train",
            variables_path=f"{APP_ML_PATH}/data/initial/variables.txt",
    ):
        super().__init__(path=path,
                         variables_path=variables_path,
                         target="ALL")
        self.target = self.variables

    def __getitem__(self, item):
        return self.features[item, :]

    def __len__(self):
        return len(self.features)


class EnergyDataset(ParticleDataset):
    _electron_label = "Truth"
    _energy_label = "p_truth_E"

    def __init__(self,
                 path=f"{APP_ML_PATH}/data/initial/train",
                 variables_path=f"{APP_ML_PATH}/data/initial/regression_variables.txt",
                 target=_energy_label):
        super().__init__(path, variables_path, target)
        is_electron = self._all_data[self._electron_label]
        self.features = self.features[is_electron == 1]
        self.target = self.target[is_electron == 1].numpy()
        self._m = np.mean(self.target)
        self._s = 2 * np.std(self.target)
        self.target = (self.target - np.mean(self.target)) / (2 * np.std(self.target))
        self.target = torch.from_numpy(self.target)

    def reverse_transform(self, x):
        return x * self._s + self._m


def main():
    d = ParticleDataset(target="ALL")
    dae = AEDataset(
        path=f"{APP_ML_PATH}/data/initial/test",
        variables_path=f"{APP_ML_PATH}/data/initial/variables.txt",
    )
    e = EnergyDataset()
    print(d[42])
    print(d.features.shape)
    print(dae[42])
    print(len(dae))
    print(e[42])
    print(len(e))


if __name__ == "__main__":
    main()
