import lightning as L
import optuna
import torch.nn.functional
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
import pickle
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from lightning.pytorch.callbacks import TQDMProgressBar

from model import LightningFullyConnected, BigLightningModel, LitAutoEncoder
from dataset import ParticleDataset, AEDataset, EnergyDataset


APP_ML_DIR = "/home/amh/Documents/Coding/GitHub/AppliedML2023"


def single_run(
        hidden_channels,
        decode_channels,
        hidden_layers,
        p_dropout,
        lr,
        activation=nn.LeakyReLU,
        final_activation=nn.Sigmoid,
        batch_size=50000,
        optimizer=optim.AdamW,
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        loss_fn=F.binary_cross_entropy,
        eval_loss=F.binary_cross_entropy,
        in_channels=15,
        out_channels=1,
        use_wandb=True,
        used_model=LightningFullyConnected,
        final_save=None,
        num_workers=8, devices=1, max_epochs=1000, patience=10, refresh_rate=10,
        dataset=ParticleDataset,
        **kwargs
):
    model = used_model(in_channels, out_channels, hidden_channels, decode_channels, hidden_layers,
                                    p_dropout, activation, final_activation, lr, batch_size, optimizer, scheduler, loss_fn)
    train_data, val_data = dataset(**kwargs).split_data(0.8)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    trainer = L.Trainer(devices=devices, accelerator="gpu",
                        max_epochs=max_epochs,
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=patience),
                                   TQDMProgressBar(refresh_rate=refresh_rate)],
                        )
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    if final_save is not None:
        trainer.save_checkpoint(final_save)
    return eval_loss(model(val_data[:][0])[:, 0], val_data[:][1])


def final_run(
        hidden_channels,
        decode_channels,
        hidden_layers,
        p_dropout,
        lr,
        activation=nn.LeakyReLU,
        final_activation=nn.Sigmoid,
        batch_size=50000,
        optimizer=optim.AdamW,
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        loss_fn=F.binary_cross_entropy,
        in_channels=15,
        out_channels=1,
        use_wandb=True,
        used_model=LightningFullyConnected,
        final_save=None,
        num_workers=8, devices=1, max_epochs=1000, patience=10, refresh_rate=10,
        dataset=ParticleDataset,
        **kwargs
):
    model = used_model(in_channels, out_channels, hidden_channels, decode_channels, hidden_layers,
                                    p_dropout, activation, final_activation, lr, batch_size, optimizer, scheduler, loss_fn)
    train_data = dataset(**kwargs)
    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)

    trainer = L.Trainer(devices=devices, accelerator="gpu",
                        max_epochs=max_epochs,
                        )
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=train_dataloader
                )
    if final_save is not None:
        trainer.save_checkpoint(final_save)


def model1_final_run():
    hidden_channels = 62
    decode_channels = 18
    hidden_layers = 4
    p_dropout = 0.0018583197040027992
    lr = 0.05845865046613747
    optimizer = optim.AdamW
    scheduler = optim.lr_scheduler.LinearLR
    final_run(hidden_channels, decode_channels, hidden_layers, p_dropout, lr,
              final_save="data/initial/nn_clf_final_opt.ckpt",
              optimizer=optimizer, scheduler=scheduler)


def model1_run():
    hidden_channels = 62
    decode_channels = 18
    hidden_layers = 4
    p_dropout =0.0018583197040027992
    lr = 0.05845865046613747
    optimizer = optim.AdamW
    scheduler = optim.lr_scheduler.LinearLR
    single_run(hidden_channels, decode_channels, hidden_layers, p_dropout, lr,
               final_save="data/initial/model1_opt.ckpt",
               optimizer=optimizer, scheduler=scheduler)


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


def ae_run(**kwargs):
    layer_shape = [160, 80, 40, 20, 15]
    p_dropout = 0.1
    use_batch_norm = True
    batch_size = 2500
    num_workers = 8
    final_save = "data/initial/ae_full.ckpt"

    model = LitAutoEncoder(layer_shape, p_dropout, use_batch_norm=use_batch_norm)
    train_data, val_data = AEDataset(**kwargs).split_data(0.8)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers)

    print(len(val_data))
    print(len(train_data) // batch_size)

    trainer = L.Trainer(devices=1, accelerator="gpu",
                        max_epochs=1000,
                        callbacks=[
                            EarlyStopping(monitor="val_loss", mode="min", patience=20),
                            TQDMProgressBar(refresh_rate=10)],
                        )
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    if final_save is not None:
        trainer.save_checkpoint(final_save)


def regression_run(
        hidden_channels,
        decode_channels,
        hidden_layers,
        p_dropout,
        lr,
        activation=nn.LeakyReLU,
        final_activation=None,
        batch_size=2500,
        optimizer=optim.AdamW,
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        loss_fn=F.mse_loss,
        eval_loss=F.mse_loss,
        in_channels=20,
        out_channels=1,
        used_model=LightningFullyConnected,
        final_save=None,
        num_workers=8, devices=1, max_epochs=100, patience=3, refresh_rate=10,
        dataset=EnergyDataset,
        **kwargs
):
    model = used_model(in_channels, out_channels, hidden_channels, decode_channels,
                       hidden_layers,
                       p_dropout, activation, final_activation, lr, batch_size,
                       optimizer, scheduler, loss_fn)

    train_data, val_data = dataset(**kwargs).split_data(0.8)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers)

    trainer = L.Trainer(devices=devices, accelerator="gpu",
                        max_epochs=max_epochs,
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min",
                                                 patience=patience),
                                   TQDMProgressBar(refresh_rate=refresh_rate)],
                        )
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    if final_save is not None:
        trainer.save_checkpoint(final_save)
    return eval_loss(model(val_data[:][0])[:, 0], val_data[:][1])


def regression_hyper(trial: optuna.Trial):
    def rel_mae(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((((input - target) / target)**2)**0.5)

    def rel_mse(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(((input - target) / target)**2)

    hidden_channels = trial.suggest_int("hidden_channels", 5, 100, log=True)
    decode_channels = trial.suggest_int("decode_channels", 5, 100, log=True)
    hidden_layers = trial.suggest_int("hidden_layers", 1, 7)
    p_dropout = trial.suggest_float("p_dropout", 0.001, 0.5, log=True)
    lr = trial.suggest_float("lr", 0.00001, 1., log=True)
    optimizer = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "Adagrad"])
    scheduler = trial.suggest_categorical("scheduler", ["CosineAnnealingLR", "LinearLR",
                                                        "ReduceLROnPlateau"])

    if optimizer == "Adam":
        u_opt = optim.Adam
    elif optimizer == "AdamW":
        u_opt = optim.AdamW
    else:
        u_opt = optim.Adagrad

    if scheduler == "CosineAnnealingLR":
        u_sch = optim.lr_scheduler.CosineAnnealingLR
    elif scheduler == "LinearLR":
        u_sch = optim.lr_scheduler.LinearLR
    else:
        u_sch = optim.lr_scheduler.ReduceLROnPlateau
    return single_run(hidden_channels, decode_channels, hidden_layers, p_dropout,
                      lr, final_activation=nn.Sigmoid,
                      optimizer=u_opt, scheduler=u_sch,
                      loss_fn=F.binary_cross_entropy, eval_loss=F.binary_cross_entropy,
                      batch_size=50000, max_epochs=100, patience=3)


def main():
    # model1_final_run()
    # model1_run()
    # model2_run()
    # ae_run()
    # study = optuna.create_study(direction="minimize")
    # optuna.logging.set_verbosity(optuna.logging.INFO)
    # study.optimize(regression_hyper, n_trials=100)
    # with open(f"{APP_ML_DIR}/data/initial/nn_clf_study.pkl", "wb") as f:
    #     pickle.dump(study, f)
    regression_run(hidden_channels=26, decode_channels=12, hidden_layers=1,
                   p_dropout=0.001154748296085491, lr=0.016811382693451612,
                   optimizer=optim.AdamW, scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                   loss_fn=F.mse_loss, batch_size=50000, max_epochs=1000, patience=10,
                   final_save=f"{APP_ML_DIR}/data/initial/reg_trained.ckpt")


if __name__ == "__main__":
    main()
