import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import stats
from torch.utils.data import Dataset


class AlephData(Dataset):
    def __init__(self, file_name="../Week1/AlephBtag_MC_train_Nev5000.csv"):

        features, labels = self._get_data(file_name)
        self.features = torch.from_numpy(features.to_numpy())
        self.labels = torch.from_numpy(labels.to_numpy())

    def __getitem__(self, item):
        return self.features[item, :], self.labels[item]

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _get_data(file_name='AlephBtag_MC_train_Nev50000.csv'):
        # Get data (with this very useful NumPy reader):
        data = np.genfromtxt(fname=file_name, names=True)  # For faster running
        # data = np.genfromtxt('AlephBtag_MC_train_Nev50000.csv', names=True)   # For more data

        # Kinematics (energy and direction) of the jet:
        energy = data['energy']
        cTheta = data['cTheta']
        phi = data['phi']

        # Classification variables (those used in Aleph's NN):
        prob_b = data['prob_b']
        spheri = data['spheri']
        pt2rel = data['pt2rel']
        multip = data['multip']
        bqvjet = data['bqvjet']
        ptlrel = data['ptlrel']
        # Aleph's NN score:
        nnbjet = data['nnbjet']
        # Truth variable whether it really was a b-jet or not (i.e. target)
        isb = data['isb']

        features = pd.DataFrame({
            "prob_b": prob_b,
            "spheri": spheri,
            "pt2rel": pt2rel,
            "multip": multip,
            "bqvjet": bqvjet,
            "ptlrel": ptlrel,
        }, dtype=np.float32)
        labels = pd.Series(isb, dtype='int16')
        return features, labels


def main():
    d = AlephData()
    print(d[42])
    pass


if __name__ == '__main__':
    main()
