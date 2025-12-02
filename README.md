# Energy Efficient STGNN

Spatio-Temporal Graph Neural Network for energy-efficient forecasting with continual learning.

## Project Structure

See directory tree for complete organization.


## Plan

Quick plan you should follow (concise)
Use Option A (recommended):
Train: first 60% of pre-COVID timeline
Val: next 10%
Test (holdout): next 10%
Continual stream: final 20%, split into monthly batches (gives ~6–12 updates depending on pre-COVID length)
Continual update policy: fine-tune on each batch for 1–5 epochs (start with 1), measure accuracy & energy per update.
Anti-forgetting: keep a tiny replay buffer (0.1–1% of train) or use EWC if you want memory-free.
Energy logging: use CodeCarbon + periodic nvidia-smi/pynvml snapshots and write all readings to CSV.
Report core comparisons: baseline full retrain vs. incremental update vs. optimized incremental (pruning/quant/FP16). Report RMSE, Forgetting, kWh/update, CO₂e, and ΔRMSE/ΔkWh.


## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```
