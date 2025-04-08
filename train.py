import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer

from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.loggers import WandbLogger
import wandb

data_path = '/Users/laurenbeede/Google Drive/My Drive/STAT5243/FinalProject/Dexcom Data/'
csv_files = glob.glob(data_path + '*.csv')
df_list = [pd.read_csv(file, skiprows=range(1, 11)) for file in csv_files]
dexcom_df = pd.concat(df_list, ignore_index=True)

# Data cleaning
dexcom_df.loc[dexcom_df['Glucose Value (mg/dL)'] == 'Low', 'Glucose Value (mg/dL)'] = 39
dexcom_df.loc[dexcom_df['Glucose Value (mg/dL)'] == 'High', 'Glucose Value (mg/dL)'] = 401
dexcom_df['Glucose Value (mg/dL)'] = dexcom_df['Glucose Value (mg/dL)'].astype(int)
dexcom_df['Timestamp'] = pd.to_datetime(dexcom_df['Timestamp (YYYY-MM-DDThh:mm:ss)'])
dates_to_remove = ['2023-11-21', '2024-02-14', '2024-08-07']
dates_to_remove = [pd.to_datetime(date).date() for date in dates_to_remove]
dexcom_df = dexcom_df[~dexcom_df['Timestamp'].dt.date.isin(dates_to_remove)]

test_df = dexcom_df.copy()
test_df['day_id'] = test_df['Timestamp'].dt.date
test_df = test_df.sort_values(['day_id', 'Timestamp'])
test_df["time_idx"] = test_df.groupby("day_id").cumcount()
test_df["minute_of_day"] = test_df["Timestamp"].dt.hour * 60 + test_df["Timestamp"].dt.minute

max_encoder_length = 288  # one day of 5-minute readings
max_prediction_length = 12

# Keep only groups (days) with at least encoder + prediction observations
data = test_df.groupby("day_id").filter(lambda x: len(x) >= max_encoder_length + max_prediction_length)

# Define training cutoff so that the tail is held out for validation
training_cutoff = data["time_idx"].max() - max_prediction_length

data["Glucose Value (mg/dL)"] = data["Glucose Value (mg/dL)"].astype(np.float32)

normalizer = GroupNormalizer(groups=["day_id"], transformation="softplus", center=False)
normalizer.fit(data["Glucose Value (mg/dL)"], data)

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Glucose Value (mg/dL)",
    group_ids=["day_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["Glucose Value (mg/dL)"],
    time_varying_known_reals=["minute_of_day"],
    target_normalizer=normalizer
)

batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)

validation = TimeSeriesDataSet.from_dataset(
    training,
    data,
    predict=True,
    stop_randomization=True
)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

def train():
    # Initialize a wandb run for this training sweep trial
    with wandb.init() as run:
        config = wandb.config
        # Create a wandb logger that will automatically log metrics and hyperparameters.
        wandb_logger = WandbLogger(project="blood-glucose", log_model="all")
        wandb_logger.log_hyperparams({
            "learning_rate": config.learning_rate,
            "hidden_size": config.hidden_size,
            "dropout": config.dropout
        })
        
        # Determine what Loss we will use
        loss_str = config.loss
        if loss_str == "RMSE":
            from pytorch_forecasting.metrics import RMSE
            loss_fn = RMSE()
            output_size_config = 1
        elif loss_str == "MAE":
            from pytorch_forecasting.metrics import MAE
            loss_fn = MAE()
            output_size_config = 1
        elif loss_str == "QuantileLoss":
            from pytorch_forecasting.metrics import QuantileLoss
            loss_fn = QuantileLoss()
            output_size_config = 7
        else:
            raise ValueError("Unsupported loss function")

        # Create the TemporalFusionTransformer model using hyperparameters from wandb.config.
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=config.learning_rate,  # use the sweep value here
            hidden_size=config.hidden_size,
            attention_head_size=1,    # you may also parameterize this if desired
            dropout=config.dropout,
            hidden_continuous_size=8,
            output_size=output_size_config,  # point forecast
            loss=loss_fn,
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        
        # Print to verify that the model is a LightningModule
        from lightning.pytorch import LightningModule
        print("Model is LightningModule:", isinstance(tft, LightningModule))
        
        trainer = Trainer(
            accelerator="mps",
            devices=1,
            max_epochs=wandb.config.epochs,  # you can adjust epochs as needed
            logger=wandb_logger
        )
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# Run the training function (this would be called by wandb agent during sweeps)
if __name__ == "__main__":
    train()
