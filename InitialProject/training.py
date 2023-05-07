import pickle

import optuna
from optuna.integration import WeightsAndBiasesCallback
from optuna.trial import TrialState

from neural_net import FullyConnectedModel
from initial_dataset import InitialDataset

import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb

from sklearn.model_selection import ParameterGrid


COUNT = 0
IN_CHANNELS = 15


class Model(L.LightningModule):
    def __init__(self,
                 loss=F.binary_cross_entropy,
                 model=FullyConnectedModel,
                 lr=0.001,
                 **kwargs):
        super().__init__()
        self.model = model(**kwargs)
        self.loss = loss
        self.lr = lr
        self.total_correct = 0
        self.best_acc = 0
        self.best_epoch = 0
        self.train_total_correct = 0
        self.total_train_data = 0
        self.total_validation_data = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, target = batch
        features = features.view(features.size(0), -1)
        target_predict = self.model(features)
        self.train_total_correct += torch.sum(torch.round(target_predict) == target)
        self.total_train_data += len(target)
        return self.loss(target_predict, target)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_predict = self.model(x)
        val_loss = self.loss(y_predict, y)
        self.log("val_loss", val_loss)
        self.total_correct += torch.sum(torch.round(y_predict) == y)
        self.total_validation_data += len(y)
        return val_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                      T_0=10,
                                                                      ),
                "monitor": "val_loss",
            }
        }

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def on_validation_epoch_end(self):
        accuracy = self.total_correct / self.total_validation_data
        self.log("val_acc", accuracy, prog_bar=True)
        if accuracy >= self.best_acc:
            self.best_acc = accuracy
            self.best_epoch = self.current_epoch
        self.total_correct = 0
        self.total_validation_data = 0

    def on_train_epoch_end(self):
        train_accuracy = self.train_total_correct / self.total_train_data
        self.log("train_acc", train_accuracy, prog_bar=True)
        self.train_total_correct = 0
        self.total_train_data = 0


def objective(trial):
    n_layers = trial.suggest_int("n_layers", 3, 10)
    hidden_channels = trial.suggest_int("hidden_channels", 5, 80, log=True)
    decode_channels = trial.suggest_int("decode_channels", 5, 40, log=True)
    p_dropout = trial.suggest_float("p_dropout", 0.01, 0.3, log=True)
    max_epochs = trial.suggest_int("max_epochs", 15, 75)
    batch_size = trial.suggest_int("batch_size", 1250, 12500, log=True)
    lr = trial.suggest_float("lr", 0.00001, 0.1, log=True)

    train_data, valid_data = InitialDataset().split_data()
    model = Model(loss=F.binary_cross_entropy,
                  in_channels=IN_CHANNELS, out_channels=1,
                  decode_channels=decode_channels,
                  hidden_channels=hidden_channels,
                  n_layers=n_layers,
                  p_dropout=p_dropout,
                  final_activation=nn.Sigmoid(),
                  lr=lr)
    trainer = L.Trainer(accelerator="gpu",
                        devices=1,
                        max_epochs=max_epochs)
    trainer.fit(model,
                DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8),
                DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=8)
                )
    return model.best_acc


def single_run(use_wandb=True, save_model_path=None, lr=0.0003, p=0.05, batch_size=2500):
    if use_wandb:
        trainer = L.Trainer(logger=WandbLogger(name=f"initial", save_dir="./wandb", project="initialProject"),
                            accelerator="gpu",
                            devices=1,
                            max_epochs=100,
                            default_root_dir=save_model_path,
                            )
    else:
        trainer = L.Trainer(accelerator="gpu",
                            devices=1,
                            max_epochs=100,
                            default_root_dir=save_model_path,
                            )

    train_data, valid_data = InitialDataset().split_data(seed=42)
    model = Model(loss=F.binary_cross_entropy,
                  in_channels=IN_CHANNELS, out_channels=1, decode_channels=25, hidden_channels=625, n_layers=1,
                  p_dropout=p, lr=lr,
                  final_activation=nn.Sigmoid())
    trainer.fit(model,
                DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8),
                DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=8))


def hyper_param_opt():
    wandb_kwargs = {"project": "my-project",
                    "name": f"initial01",
                    }
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=False)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25, callbacks=[wandbc])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open("/home/amh/Documents/coding/GitHub/AppliedML2023/data/initial/nn_study.pickle", "wb") as handle:
        pickle.dump(study)


def main():
    # hyper_param_opt()
    create_soup()
    # single_run(use_wandb=False)


def create_soup():
    soup_path = "/home/amh/Documents/coding/GitHub/AppliedML2023/data/initial/SoupIngredients"
    for index, params in enumerate(ParameterGrid({"lr": [0.01, 0.03, 0.001, 0.003, 0.0001],
                                                  "p": [0.05, 0.1, 0.2, 0.3, 0.5],
                                                  "batch_size": [1250, 2500, 5000, 7500, 10000]})):
        single_run(use_wandb=False, save_model_path=f"{soup_path}/ingredient{index}.pth", **params)


if __name__ == "__main__":
    main()
