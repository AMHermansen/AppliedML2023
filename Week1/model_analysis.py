import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from time import perf_counter
import tqdm


class AlephDataset(Dataset):
    def __init__(self, file_path='AlephBtag_MC_train_Nev50000.csv', device='cuda'):
        self.device = device
        # Get data (with this very useful NumPy reader):
        data = np.genfromtxt(file_path, names=True)  # For faster running
        # data = np.genfromtxt('AlephBtag_MC_train_Nev50000.csv', names=True)   # For more data

        # Kinematics (energy and direction) of the jet:
        energy = data['energy']
        cTheta = data['cTheta']
        phi    = data['phi']

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
        isb    = data['isb']

        features = pd.DataFrame({
            "prob_b": prob_b,
            "spheri": spheri,
            "pt2rel": pt2rel,
            "multip": multip,
            "bqvjet": bqvjet,
            "ptlrel": ptlrel,
        })

        labels = pd.Series(isb, dtype='int16')
        self.features = features.to_numpy()
        self.features = (self.features - np.mean(self.features, axis=0)) / np.std(self.features, axis=0)
        self.labels = labels.to_numpy(dtype=np.int8)
        self.labels = np.stack([self.labels, (1 - self.labels)])
        self.labels = self.labels.T
        self.labels.flatten('C')

        self.features = torch.Tensor(self.features).to(self.device)
        self.labels = torch.Tensor(self.labels).to(self.device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx]


class FullyConnectedBlock(nn.Module):
    def __init__(self, in_channels, out_channels=15, p_dropout=0.25, activation=nn.LeakyReLU()):
        super().__init__()
        layer = nn.Linear(in_channels, out_channels)
        drop = nn.Dropout(p=p_dropout)
        activation = activation
        self.model = nn.Sequential(layer, drop, activation)
        self.skip_connection = in_channels == out_channels


    def forward(self, x):
        out = self.model(x)
        if self.skip_connection:
            out += x
        return out


class FirstModel(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=15, decode_channels=6, out_channels=2, n_layers=4, p_dropout=0.25, activation=nn.LeakyReLU()):
        super().__init__()
        layers = [nn.Linear(in_channels, hidden_channels), nn.LeakyReLU()]
        for _ in range(n_layers):
            layers.append(FullyConnectedBlock(hidden_channels, hidden_channels, p_dropout=p_dropout, activation=activation))
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Sequential(*[nn.Linear(hidden_channels, decode_channels),
                                       nn.LeakyReLU(),
                                       nn.Linear(decode_channels, out_channels),
                                       nn.Softmax(1)])

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


def train_one_epoch(epoch_index, tb_writer, loss_fn, model, optimizer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


if __name__ == '__main__':
    model = FirstModel(hidden_channels=25, n_layers=4).to(torch.device('cuda'))

    aleph_train_data = AlephDataset('../data/AlephBtag_MC_train_Nev5000000.csv', device='cuda')
    aleph_valid_data = AlephDataset('AlephBtag_MC_train_Nev50000.csv', device='cuda')

    VALID_BATCH_SIZE = 50000
    TRAIN_BATCH_SIZE = 250000
    training_loader = DataLoader(aleph_train_data, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    validation_loader = DataLoader(aleph_valid_data, shuffle=False, batch_size=VALID_BATCH_SIZE)

    NEW_MODEL = True
    if NEW_MODEL:
        model = FirstModel(hidden_channels=25, n_layers=6, p_dropout=0.05, activation=nn.LeakyReLU())
    model.to(device=torch.device('cuda:0'))
    print("Starting")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    EPOCHS = 25

    best_vloss = 1_000_000.

    Path("models").mkdir(parents=True, exist_ok=True)
    for epoch in range(EPOCHS):
        start_time = perf_counter()
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, loss_fn, model=model, optimizer=optimizer)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        running_acc = 0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
            running_acc += torch.sum(torch.argmax(voutputs, axis=1) == torch.argmax(vlabels, axis=1))
        running_acc = running_acc / ((i + 1) * VALID_BATCH_SIZE)

        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss} Validation accuracy: {running_acc}')

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'models/model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
        end_time = perf_counter()
        print(f"Epoch time: {end_time-start_time}s")
        epoch_number += 1
