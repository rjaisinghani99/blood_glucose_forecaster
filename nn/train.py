import pandas as pd
import glob
import numpy as np
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer, RecurrentNetwork, Baseline
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb
import torch
import pickle

# Load and clean data
data_path = '/Users/laurenbeede/Google Drive/My Drive/STAT5243/FinalProject/Dexcom Data/'
csv_files = glob.glob(data_path + '*.csv')
df_list = [pd.read_csv(file, skiprows=range(1, 11)) for file in csv_files]
dexcom_df = pd.concat(df_list, ignore_index=True)

dexcom_df.loc[dexcom_df['Glucose Value (mg/dL)'] == 'Low', 'Glucose Value (mg/dL)'] = 39
dexcom_df.loc[dexcom_df['Glucose Value (mg/dL)'] == 'High', 'Glucose Value (mg/dL)'] = 401
dexcom_df['Glucose Value (mg/dL)'] = dexcom_df['Glucose Value (mg/dL)'].astype(int)
dexcom_df['Timestamp'] = pd.to_datetime(dexcom_df['Timestamp (YYYY-MM-DDThh:mm:ss)'])

# Remove known corrupted dates
dates_to_remove = ['2023-11-21', '2024-02-14', '2024-08-07']
dates_to_remove = [pd.to_datetime(date).date() for date in dates_to_remove]
dexcom_df = dexcom_df[~dexcom_df['Timestamp'].dt.date.isin(dates_to_remove)]

# Create time-based columns
dexcom_df['day_id'] = dexcom_df['Timestamp'].dt.date
dexcom_df = dexcom_df.sort_values(['day_id', 'Timestamp'])

# Filter for days with exactly 288 readings (i.e., one per 5 minutes)
dexcom_df = dexcom_df.groupby("day_id").filter(lambda x: len(x) == 288).copy()

# Recompute time_idx after filtering
dexcom_df['time_idx'] = dexcom_df.groupby("day_id").cumcount()
dexcom_df['minute_of_day'] = dexcom_df['Timestamp'].dt.hour * 60 + dexcom_df['Timestamp'].dt.minute

max_encoder_length = 276
max_prediction_length = 12

# Split train/val/test
unique_days = sorted(dexcom_df['day_id'].unique())
n_days = len(unique_days)

# Indices for splits
train_end = int(n_days * 0.75)
val_end = int(n_days * 0.90)  # 75% + 15%

# Assign splits
train_days = unique_days[:train_end]
val_days = unique_days[train_end:val_end]


train_data = dexcom_df[dexcom_df['day_id'].isin(train_days)].copy()
val_data = dexcom_df[dexcom_df['day_id'].isin(val_days)].copy()


train_data['Glucose Value (mg/dL)'] = train_data['Glucose Value (mg/dL)'].astype(np.float32)
val_data['Glucose Value (mg/dL)'] = val_data['Glucose Value (mg/dL)'].astype(np.float32)

# Normalize target
combined_for_normalizer = pd.concat([train_data, val_data])
normalizer = GroupNormalizer(groups=["day_id"], transformation="softplus", center=False)
normalizer.fit(combined_for_normalizer["Glucose Value (mg/dL)"], combined_for_normalizer)

# Build TimeSeriesDataSet objects
training = TimeSeriesDataSet(
    train_data,
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
    val_data,
    predict=False,
    stop_randomization=True,
    categorical_encoders={"group_ids": {"add_nan": True}}

)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

def train(normalizer):
    with wandb.init() as run:
        job_id = wandb.util.generate_id()
        config = wandb.config

        wandb_logger = WandbLogger(project="blood-glucose-v2", name=f"run_{job_id}", log_model="all")
        if config.model != "Baseline":
            wandb_logger.log_hyperparams({
                "learning_rate": config.learning_rate,
                "hidden_size": config.hidden_size,
                "dropout": config.dropout
            })
        else:
            wandb_logger.log_hyperparams({"model": "Baseline"})

        # Save normalizer with run-specific name
        normalizer_path = f"trained_normalizer_{run.id}.pkl"
        with open(normalizer_path, "wb") as f:
            pickle.dump(normalizer, f)

        # Log as a unique artifact per run
        artifact = wandb.Artifact(f"normalizer-{run.id}", type="preprocessing")
        artifact.add_file(normalizer_path)
        run.log_artifact(artifact)


        if config.model == "Baseline":
            model = Baseline()
            loss_fn = None  # Not used
            output_size = 1
        else:
            if config.loss == "RMSE":
                from pytorch_forecasting.metrics import RMSE
                loss_fn = RMSE()
                output_size = 1
            elif config.loss == "MAE":
                from pytorch_forecasting.metrics import MAE
                loss_fn = MAE()
                output_size = 1
            elif config.loss == "QuantileLoss":
                from pytorch_forecasting.metrics import QuantileLoss
                loss_fn = QuantileLoss(config.quantiles)
                output_size = len(loss_fn.quantiles)
            else:
                raise ValueError("Unsupported loss function")


        early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=True)
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename="best")

        if config.model == "TemporalFusionTransformer":
            # Create the TemporalFusionTransformer model using hyperparameters from wandb.config.
             model = TemporalFusionTransformer.from_dataset(
                        training,
                        learning_rate=config.learning_rate,
                        hidden_size=config.hidden_size,
                        attention_head_size=config.attention_head_size,
                        dropout=config.dropout,
                        hidden_continuous_size=config.hidden_continuous_size,
                        output_size=output_size,
                        loss=loss_fn,
                        log_interval=10,
                        reduce_on_plateau_patience=4,
                        weight_decay=config.weight_decay
                    )
             trainer = Trainer(
                accelerator="mps",
                devices=1,
                max_epochs=config.epochs,
                logger=wandb_logger,
                callbacks=[early_stop_callback, checkpoint_callback]
            )
        elif config.model == "LSTM":
            # Create the TemporalFusionTransformer model using hyperparameters from wandb.config.
            model = RecurrentNetwork.from_dataset(
                training,
                learning_rate=config.learning_rate,  # use the sweep value here
                hidden_size=config.hidden_size,
                dropout=config.dropout,
                cell_type="LSTM", 
                loss=loss_fn,
                output_size=output_size,
                log_interval=10,
                reduce_on_plateau_patience=4,
                weight_decay=config.weight_decay
            )
            trainer = Trainer(
                accelerator="mps",
                devices=1,
                max_epochs=config.epochs,
                logger=wandb_logger,
                callbacks=[early_stop_callback, checkpoint_callback]
            )
        elif config.model == "Baseline":
            all_predictions = []
            all_actuals = []

            for x, y in val_dataloader:
                preds = model(x)["prediction"]  # shape: [B, T, 1]
                all_predictions.append(preds.detach())
                all_actuals.append(y[0].detach())  # [B, T]

            predictions = torch.cat(all_predictions, dim=0)  # [B, T, 1]
            actuals = torch.cat(all_actuals, dim=0)  # [B, T]

            from pytorch_forecasting.metrics import SMAPE, RMSE, MAE, QuantileLoss

            if config.loss == "QuantileLoss":
                quantiles = getattr(config, "quantiles", [0.2, 0.5, 0.8])
                num_q = len(quantiles)

                # Expand predictions to shape [B, T, Q]
                predictions = predictions.unsqueeze(-1) 
                predictions = predictions.expand(-1, -1, num_q)  # keep batch and time the same, expand to Q

                metric = QuantileLoss(quantiles=quantiles)

            else:
                metrics_dict = {"SMAPE": SMAPE(), "RMSE": RMSE(), "MAE": MAE()}
                metric = metrics_dict[config.loss]

            val_loss = metric(predictions, actuals)

            print(f"Baseline {config.loss} on validation set: {val_loss.item()}")
            wandb.log({f"val_{config.loss.lower()}": val_loss.item()})
            return

        
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == "__main__":
    train(normalizer)