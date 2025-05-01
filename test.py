import pandas as pd
import numpy as np
import glob
import torch
import wandb
import pickle
import argparse
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer
)

# --- Parse CLI arguments ---
parser = argparse.ArgumentParser(description="Run TFT test prediction using W&B artifact")
parser.add_argument(
    "--artifact_name",
    type=str,
    required=True,
    help="Name of W&B artifact to use, e.g., 'User/blood-glucose/model-best:v0'"
)
args = parser.parse_args()

# --- Config ---
data_path = "/Users/laurenbeede/Google Drive/My Drive/STAT5243/FinalProject/Dexcom Data/"
max_encoder_length = 276
max_prediction_length = 12
batch_size = 64

# --- Load and preprocess data ---
csv_files = glob.glob(data_path + '*.csv')
df_list = [pd.read_csv(file, skiprows=range(1, 11)) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

df.loc[df['Glucose Value (mg/dL)'] == 'Low', 'Glucose Value (mg/dL)'] = 39
df.loc[df['Glucose Value (mg/dL)'] == 'High', 'Glucose Value (mg/dL)'] = 401
df['Glucose Value (mg/dL)'] = df['Glucose Value (mg/dL)'].astype(float)
df['Timestamp'] = pd.to_datetime(df['Timestamp (YYYY-MM-DDThh:mm:ss)'])

dates_to_remove = ['2023-11-21', '2024-02-14', '2024-08-07']
dates_to_remove = [pd.to_datetime(d).date() for d in dates_to_remove]
df = df[~df['Timestamp'].dt.date.isin(dates_to_remove)]

# Create time-based columns
df['day_id'] = df['Timestamp'].dt.date
df = df.sort_values(['day_id', 'Timestamp'])

# Filter for days with exactly 288 readings
df = df.groupby("day_id").filter(lambda x: len(x) == 288).copy()

# Recompute time_idx after filterin
df['time_idx'] = df.groupby("day_id").cumcount()
df['minute_of_day'] = df['Timestamp'].dt.hour * 60 + df['Timestamp'].dt.minute

# --- Get test days (last 10%) ---
unique_days = sorted(df['day_id'].unique())
n_days = len(unique_days)
val_end = int(n_days * 0.90)
test_days = unique_days[val_end:]
test_data = df[df['day_id'].isin(test_days)].copy()

# --- Load model from W&B artifact ---
wandb.login()
run = wandb.init(project="blood-glucose-v2", job_type="testing")
artifact = run.use_artifact(args.artifact_name, type="model")
artifact_dir = artifact.download()
model = TemporalFusionTransformer.load_from_checkpoint(f"{artifact_dir}/model.ckpt")
model.to("mps")  # or "cuda" if on GPU

# Download associated normalizer artifact
normalizer_artifact_name = args.artifact_name.replace("model", "normalizer").split(":")[0]  # drop version if needed
normalizer_artifact = run.use_artifact(normalizer_artifact_name + ":latest", type="preprocessing")
normalizer_path = normalizer_artifact.download()

# --- Load trained normalizer --- 
artifact_id = artifact.name.split("/")[-1].split(":")[0].replace("model-", "")
with open(f"{normalizer_path}/trained_normalizer_{artifact_id}.pkl", "rb") as f:
    normalizer = pickle.load(f)

# --- Create reference dataset using full DataFrame (same structure as training) ---
reference_dataset = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="Glucose Value (mg/dL)",
    group_ids=["day_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["Glucose Value (mg/dL)"],
    time_varying_known_reals=["minute_of_day"],
    target_normalizer=normalizer
)

# --- Build test dataset using training structure ---
test_dataset = TimeSeriesDataSet.from_dataset(
    reference_dataset,
    test_data,
    predict=False,
    stop_randomization=True
)
test_dataloader = test_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# --- Predict ---
preds = model.predict(test_dataloader)
preds = preds.detach().cpu().numpy().flatten()

# --- Get true values ---
actuals = []
for batch in test_dataloader:
    x, y = batch
    actuals.append(y[0].detach().cpu().numpy())
actuals = np.concatenate(actuals, axis=0).flatten()

# --- Save results ---
results_df = pd.DataFrame({
    "true_glucose": actuals,
    "predicted_glucose": preds
})
results_df.to_csv("test_predictions_with_truth.csv", index=False)
