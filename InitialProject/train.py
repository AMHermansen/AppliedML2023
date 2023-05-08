import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from model import LightningFullyConnected
from dataset import ParticleDataset
from argparse import ArgumentParser
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from lightning.pytorch.callbacks import TQDMProgressBar


def single_run(
        hidden_channels,
        decode_channels,
        hidden_layers,
        p_dropout,
        lr,
        activation=nn.LeakyReLU,
        final_activation=nn.Sigmoid,
        batch_size=2500,
        optimizer=optim.AdamW,
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        loss_fn=F.binary_cross_entropy,
        in_channels=15,
        out_channels=1,
        use_wandb=True,
):
    model = LightningFullyConnected(in_channels, out_channels, hidden_channels, decode_channels, hidden_layers,
                                    p_dropout, activation, final_activation, lr, batch_size, optimizer, scheduler, loss_fn)
    train_data, val_data = ParticleDataset().split_data(0.8)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=8)

    trainer = L.Trainer(devices=1, accelerator="gpu",
                        max_epochs=200,
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min"),
                                   TQDMProgressBar(refresh_rate=10)],
                        )
    trainer.fit(model=model,)


def main():
    hidden_channels = 40
    decode_channels = 6
    hidden_layers = 5
    p_dropout = 0.1
    lr = 0.0003
    single_run(hidden_channels, decode_channels, hidden_layers, p_dropout, lr)


if __name__ == "__main__":
    main()
