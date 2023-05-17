import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import lightning as L
from dataset import ParticleDataset


class FullyConnectedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, p_dropout, activation=nn.LeakyReLU, skip_connection=True):
        if (in_channels != out_channels) and skip_connection:
            raise ValueError("In and out channels must be identical to use skip connections")
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Dropout(p_dropout),
            activation(),
        )
        self.skip_connection = skip_connection

    def forward(self, x):
        out = self.model(x)
        if self.skip_connection:
            out += x
        return out


class FullyConnectedModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, decode_channels, hidden_layers,
                 p_dropout, activation=nn.LeakyReLU, final_activation=None):
        super().__init__()
        used_activation = activation()
        layers = [nn.Linear(in_channels, hidden_channels), used_activation]
        for _ in range(hidden_layers):
            layers.append(FullyConnectedBlock(hidden_channels, hidden_channels, p_dropout, activation))
        self.encoder = nn.Sequential(*layers)
        if final_activation is not None:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_channels, decode_channels),
                used_activation,
                nn.Linear(decode_channels, out_channels),
                final_activation()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_channels, decode_channels),
                used_activation,
                nn.Linear(decode_channels, out_channels),
            )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class LightningFullyConnected(L.LightningModule):
    def __init__(self, in_channels, out_channels, hidden_channels, decode_channels, hidden_layers,
                 p_dropout, activation=nn.LeakyReLU, final_activation=None,
                 lr=0.0003, batch_size=2500, optimizer=optim.AdamW, scheduler=optim.lr_scheduler.CosineAnnealingLR,
                 loss_fn=F.binary_cross_entropy):
        super().__init__()
        self.model = FullyConnectedModel(in_channels, out_channels, hidden_channels, decode_channels,
                                         hidden_layers, p_dropout, activation, final_activation)

        self._reset_count()
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, target = batch
        target_pred = self.model(features)[:, 0]
        loss = self.loss_fn(target_pred, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, target = batch
        target_pred = self.model(features)[:, 0]
        val_loss = self.loss_fn(target_pred, target)
        self._val_total += len(target)
        self._val_correct += torch.sum(torch.round(target_pred) == target)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def on_validation_epoch_end(self):
        val_accuracy = self._val_correct / self._val_total
        self.log("val_acc", val_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self._reset_count()

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=8, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_data, num_workers=8, shuffle=False, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        lr_scheduler = self.scheduler(optimizer, T_max=10)
        return [optimizer], [{"scheduler": lr_scheduler}]

    def _reset_count(self):
        self._val_correct = 0
        self._val_total = 0


class BigLightningModel(LightningFullyConnected):
    out_size = 160

    def __init__(self, in_channels, out_channels, hidden_channels, decode_channels, hidden_layers,
                 p_dropout, activation=nn.LeakyReLU, final_activation=None,
                 lr=0.0003, batch_size=2500, optimizer=optim.AdamW, scheduler=optim.lr_scheduler.CosineAnnealingLR,
                 loss_fn=F.mse_loss):
        super().__init__(in_channels, out_channels, hidden_channels, decode_channels,
                         hidden_layers, p_dropout, activation, final_activation,
                         lr, batch_size, optimizer, scheduler, loss_fn)

    def forward(self, x):
        out = self.model(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, target = batch
        target_pred = self(features)
        loss = self.loss_fn(target_pred, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, target = batch
        target_pred = self.forward(features)
        val_loss = self.loss_fn(target_pred, target)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def _reset_count(self):
        pass

    def on_validation_epoch_end(self):
        return L.LightningModule.on_validation_epoch_end(self)
