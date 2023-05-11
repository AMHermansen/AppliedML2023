import lightning as L
import torch.nn.functional
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from model import LightningFullyConnected, BigLightningModel
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
        used_model=LightningFullyConnected,
        final_save=None,
        **kwargs
):
    model = used_model(in_channels, out_channels, hidden_channels, decode_channels, hidden_layers,
                                    p_dropout, activation, final_activation, lr, batch_size, optimizer, scheduler, loss_fn)
    train_data, val_data = ParticleDataset(**kwargs).split_data(0.8)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)

    trainer = L.Trainer(devices=1, accelerator="gpu",
                        max_epochs=1000,
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10),
                                   TQDMProgressBar(refresh_rate=10)],
                        )
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    if final_save is not None:
        trainer.save_checkpoint(final_save)


def model1_run():
    hidden_channels = 20
    decode_channels = 6
    hidden_layers = 10
    p_dropout = 0.1
    lr = 0.0005
    single_run(hidden_channels, decode_channels, hidden_layers, p_dropout, lr,
               final_save="data/initial/model1.ckpt")


def model2_run():
    hidden_channels = 60
    decode_channels = 120
    out_channels = 160
    hidden_layers = 20
    p_dropout = 0.2
    lr = 0.0001
    single_run(hidden_channels, decode_channels, hidden_layers, p_dropout, lr,
               out_channels=out_channels, loss_fn=F.mse_loss,
               used_model=BigLightningModel, final_activation=None, target="ALL",
               final_save="data/initial/modelbig1.ckpt")


def main():
    # model1_run()
    model2_run()


if __name__ == "__main__":
    main()
