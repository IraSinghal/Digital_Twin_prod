
# """
# Compressor Stability Forecasting — Multi-Output Model
# Predicts:
#   1. Time to stability
#   2. Future sensor readings (+10 steps)

# Author: Updated implementation
# """

# import argparse
# import warnings
# import pickle
# import numpy as np
# import pandas as pd
# import optuna
# import time

# warnings.filterwarnings("ignore")
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GroupKFold
# from sklearn.metrics import mean_absolute_error
# from sklearn.multioutput import MultiOutputRegressor

# # =========================
# # ADD THIS HELPER AT TOP
# # =========================
# def log(msg):
#     print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# # ============================================================
# # CONFIG
# # ============================================================
# SENSORS = [
#     "airend_discharge_temp_c",
#     "delivery_pressure_kg_cm2g",
#     "fad_cfm",
#     "package_input_power_kw",
#     "spc_kw_per_m3_min",
#     "power_factor",
# ]

# FUTURE_HORIZON = 10

# N_AUG_RUNS = 20
# OPTUNA_TRIALS = 15
# CV_FOLDS = 5
# OPTUNA_CV_FOLDS = 3

# SPEED_RANGE = (0.7, 1.4)
# NOISE_SCALE_RNG = (0.2, 0.5)

# RANDOM_SEED = 42


# # ============================================================
# # FEATURE ENGINEERING
# # ============================================================
# def engineer_features_batch(run_df, stable_ref, stable_std):
#     rows = []

#     w = run_df[SENSORS].copy()
#     t_vals = run_df["elapsed_time_min"].values

#     roll20_std = w.rolling(20, min_periods=1).std()
#     roll10_std = w.rolling(10, min_periods=1).std()
#     roll20_mean = w.rolling(20, min_periods=1).mean()

#     for i in range(20, len(run_df)):
#         feats = {}

#         for s in SENSORS:
#             current_val = w[s].iloc[i]

#             feats[f"{s}_cur"] = current_val
#             feats[f"{s}_dev"] = (current_val - stable_ref[s]) / stable_std[s]
#             feats[f"{s}_roll20_std"] = roll20_std[s].iloc[i]
#             feats[f"{s}_roll10_std"] = roll10_std[s].iloc[i]

#             vals20 = w[s].iloc[max(0, i - 19): i + 1].values
#             vals10 = w[s].iloc[max(0, i - 9): i + 1].values

#             feats[f"{s}_slope20"] = np.polyfit(np.arange(len(vals20)), vals20, 1)[0] if len(vals20) >= 2 else 0.0
#             feats[f"{s}_slope10"] = np.polyfit(np.arange(len(vals10)), vals10, 1)[0] if len(vals10) >= 2 else 0.0

#         devs_current = np.array([
#             (w[s].iloc[i] - stable_ref[s]) / stable_std[s]
#             for s in SENSORS
#         ])

#         devs_roll = np.array([
#             (roll20_mean[s].iloc[i] - stable_ref[s]) / stable_std[s]
#             for s in SENSORS
#         ])

#         feats["total_dev_roll20"] = np.abs(devs_current).mean()
#         feats["total_dev_roll10"] = np.abs(devs_roll).mean()
#         feats["obs_time_min"] = t_vals[i]

#         rows.append(feats)

#     return rows


# # ============================================================
# # DATA AUGMENTATION + TARGET CREATION
# # ============================================================
# def augment_runs(df, stable_ref, stable_std, stable_time, n_runs=N_AUG_RUNS):
#     np.random.seed(RANDOM_SEED)

#     all_X, all_y, all_groups = [], [], []

#     log(f"[3/6] Generating {n_runs} augmented runs...")

#     for run_id in range(n_runs):
#         speed = np.random.uniform(*SPEED_RANGE)
#         noise_scale = np.random.uniform(*NOISE_SCALE_RNG)

#         log(f"  Run {run_id+1}/{n_runs} | Speed={speed:.2f} | Noise={noise_scale:.2f}")

#         aug = df.copy()
#         aug["elapsed_time_min"] /= speed

#         for s in SENSORS:
#             aug[s] += np.random.normal(0, stable_std[s] * noise_scale, len(aug))

#         aug_stable = stable_time / speed

#         rows = engineer_features_batch(aug, stable_ref, stable_std)
#         t_vals = aug["elapsed_time_min"].values

#         targets = []

#         for i in range(20, len(aug)):
#             t_now = t_vals[i]
#             time_target = max(aug_stable - t_now, 0)

#             future_idx = min(i + FUTURE_HORIZON, len(aug) - 1)
#             sensor_future = [aug[s].iloc[future_idx] for s in SENSORS]

#             targets.append([time_target] + sensor_future)

#         all_X.extend(rows)
#         all_y.extend(targets)
#         all_groups.extend([run_id] * len(rows))

#     log("Augmentation complete")

#     return pd.DataFrame(all_X), np.array(all_y), np.array(all_groups)



# # ============================================================
# # OPTUNA OBJECTIVE
# # ============================================================
# def make_objective(Xs, y, groups):
#     gkf = GroupKFold(n_splits=OPTUNA_CV_FOLDS)

#     def objective(trial):
#         log(f"\n[Optuna] Trial {trial.number + 1} started")

#         params = {
#             "max_iter": trial.suggest_int("max_iter", 100, 300),
#             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
#             "max_depth": trial.suggest_int("max_depth", 2, 5),
#             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 40),
#             "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 2.0),
#             "max_bins": trial.suggest_int("max_bins", 64, 255),
#         }

#         maes = []

#         for fold_num, (tr, vl) in enumerate(gkf.split(Xs, y, groups), start=1):
#             log(f"  Fold {fold_num}/{OPTUNA_CV_FOLDS} running...")

#             model = MultiOutputRegressor(
#                 HistGradientBoostingRegressor(**params, random_state=RANDOM_SEED)
#             )

#             model.fit(Xs[tr], y[tr])
#             preds = model.predict(Xs[vl])

#             mae = mean_absolute_error(y[vl][:, 0], preds[:, 0])
#             log(f"    Fold MAE: {mae:.3f}")

#             maes.append(mae)

#         mean_mae = np.mean(maes)
#         log(f"  Trial {trial.number + 1} Mean MAE: {mean_mae:.3f}")

#         return mean_mae

#     return objective


# # ============================================================
# # TRAINING PIPELINE
# # ============================================================
# def train(data_path, output_path="model.pkl"):
#     start = time.time()

#     log("=" * 60)
#     log("Compressor Stability — Multi-Output Training")
#     log("=" * 60)

#     # -----------------------------
#     log("[1/6] Loading data...")
#     df = pd.read_excel(data_path)
#     df = df.sort_values("elapsed_time_min").reset_index(drop=True)
#     log(f"Rows: {len(df)}")

#     # -----------------------------
#     log("[2/6] Computing stable reference...")
#     phase3 = df[df["phase"] == "Phase3_StableRated"]

#     STABLE_REF = phase3[SENSORS].mean()
#     STABLE_STD = phase3[SENSORS].std().replace(0, 1e-6)
#     STABLE_TIME = phase3["elapsed_time_min"].min()

#     log(f"Stable time starts at: {STABLE_TIME:.2f} min")

#     # -----------------------------
#     X, y, groups = augment_runs(df, STABLE_REF, STABLE_STD, STABLE_TIME)

#     log(f"Feature shape: {X.shape}")
#     log(f"Target shape: {y.shape}")

#     # -----------------------------
#     log("[4/6] Scaling features...")
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)
#     log("Scaling complete")

#     # -----------------------------
#     log(f"[5/6] Running Optuna ({OPTUNA_TRIALS} trials)...")

#     study = optuna.create_study(direction="minimize")
#     study.optimize(make_objective(Xs, y, groups), n_trials=OPTUNA_TRIALS)

#     best_params = study.best_params

#     log("Best parameters found:")
#     for k, v in best_params.items():
#         log(f"  {k}: {v}")

#     # -----------------------------
#     log("[6/6] Training final model...")

#     model = MultiOutputRegressor(
#         HistGradientBoostingRegressor(**best_params, random_state=RANDOM_SEED)
#     )

#     log("Fitting model (this may take time)...")
#     model.fit(Xs, y)
#     log("Model training complete")

#     # -----------------------------
#     log("Running cross-validation...")

#     gkf = GroupKFold(n_splits=CV_FOLDS)
#     maes = []

#     for fold_num, (tr, vl) in enumerate(gkf.split(Xs, y, groups), start=1):
#         log(f"  CV Fold {fold_num}/{CV_FOLDS}")

#         model.fit(Xs[tr], y[tr])
#         preds = model.predict(Xs[vl])

#         mae = mean_absolute_error(y[vl][:, 0], preds[:, 0])
#         log(f"    MAE: {mae:.3f}")

#         maes.append(mae)

#     # -----------------------------
#     bundle = {
#         "model": model,
#         "scaler": scaler,
#         "features": list(X.columns),
#         "sensors": SENSORS,
#         "future_horizon": FUTURE_HORIZON,
#         "stable_ref": STABLE_REF,
#         "stable_std": STABLE_STD,
#         "stable_time": STABLE_TIME,
#         "cv_mae": float(np.mean(maes)),
#     }

#     with open(output_path, "wb") as f:
#         pickle.dump(bundle, f)

#     log("=" * 60)
#     log(f"Training complete | CV MAE: {np.mean(maes):.2f}")
#     log(f"Saved to: {output_path}")
#     log(f"Total time: {(time.time()-start)/60:.2f} min")
#     log("=" * 60)

#     return bundle


# # ============================================================
# # INFERENCE
# # ============================================================
# def predict(bundle, X_live):
#     X_scaled = bundle["scaler"].transform(X_live)
#     pred = bundle["model"].predict(X_scaled)

#     time_pred = pred[:, 0]
#     sensor_preds = pred[:, 1:]

#     return {
#         "time_to_stability": float(time_pred[0]),
#         "sensor_forecast": dict(zip(bundle["sensors"], sensor_preds[0]))
#     }


# # ============================================================
# # MAIN
# # ============================================================
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()

# #     parser.add_argument("--data", default=r"C:\Users\Admin\OneDrive - C4i4 Lab, Samarth Udyog Technology Forum,\Attachments\Office work\DigitalTwin2\data\KES22_8p5_synthetic_test_data.xlsx")
# #     parser.add_argument("--output", default=r"C:\Users\Admin\OneDrive - C4i4 Lab, Samarth Udyog Technology Forum,\Attachments\Office work\DigitalTwin2\model.pkl")

# #     args = parser.parse_args()

# #     train(args.data, args.output)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--data",
#         default=r'C:\Users\Admin\OneDrive - C4i4 Lab, Samarth Udyog Technology Forum,\Attachments\Office work\DigitalTwin2\data\KES22_8p5_synthetic_test_data.xlsx'
#     )

#     parser.add_argument(
#         "--output",
#         default=r"C:\Users\Admin\OneDrive - C4i4 Lab, Samarth Udyog Technology Forum,\Attachments\Office work\DigitalTwin2\data\quantile_models.pkl"
#     )

#     args = parser.parse_args()

#     train(args.data, args.output)






"""
=============================================================
Compressor Stability Forecasting — Training Script
KES22 8.5 bar Series | Test Time Optimization
=============================================================
Usage:
    python compressor_train.py --data KES22_8p5_synthetic_test_data.xlsx

Outputs:
    model.pkl  — bundle containing:
                 model, scaler, features, sensors, stable_ref,
                 stable_std, stable_time, future_horizon, cv_mae

Requirements:
    pip install pandas numpy scikit-learn openpyxl optuna
=============================================================
"""

import argparse
import warnings
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import optuna


_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_TRAIN_DATA = _SCRIPT_DIR / "data" / "KES22_8p5_synthetic_test_data.xlsx"
_DEFAULT_MODEL_OUT = _SCRIPT_DIR / "data" / "quantile_models.pkl"

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error


# ============================================================
# HELPER
# ============================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# CONFIG
# ============================================================
SENSORS = [
    "airend_discharge_temp_c",
    "delivery_pressure_kg_cm2g",
    "fad_cfm",
    "package_input_power_kw",
    "spc_kw_per_m3_min",
    "power_factor",
]

FUTURE_HORIZON  = 10   # steps ahead for sensor forecast target
N_AUG_RUNS      = 20
OPTUNA_TRIALS   = 15
CV_FOLDS        = 5
OPTUNA_CV_FOLDS = 3
SPEED_RANGE     = (0.7, 1.4)
NOISE_SCALE_RNG = (0.2, 0.5)
RANDOM_SEED     = 42


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def engineer_features_batch(run_df, stable_ref, stable_std):
    """
    Build one feature-dict per valid timestep (i >= 20).
    Returns a list of dicts suitable for pd.DataFrame().
    """
    rows        = []
    w           = run_df[SENSORS].copy()
    t_vals      = run_df["elapsed_time_min"].values
    roll20_std  = w.rolling(20, min_periods=1).std()
    roll10_std  = w.rolling(10, min_periods=1).std()
    roll20_mean = w.rolling(20, min_periods=1).mean()

    for i in range(20, len(run_df)):
        feats = {}

        for s in SENSORS:
            cur = w[s].iloc[i]
            feats[f"{s}_cur"]        = cur
            feats[f"{s}_dev"]        = (cur - stable_ref[s]) / stable_std[s]
            feats[f"{s}_roll20_std"] = roll20_std[s].iloc[i]
            feats[f"{s}_roll10_std"] = roll10_std[s].iloc[i]

            vals20 = w[s].iloc[max(0, i - 19): i + 1].values
            vals10 = w[s].iloc[max(0, i - 9):  i + 1].values

            feats[f"{s}_slope20"] = (
                np.polyfit(np.arange(len(vals20)), vals20, 1)[0]
                if len(vals20) >= 2 else 0.0
            )
            feats[f"{s}_slope10"] = (
                np.polyfit(np.arange(len(vals10)), vals10, 1)[0]
                if len(vals10) >= 2 else 0.0
            )

        devs_cur  = np.array([(w[s].iloc[i] - stable_ref[s]) / stable_std[s] for s in SENSORS])
        devs_roll = np.array([(roll20_mean[s].iloc[i] - stable_ref[s]) / stable_std[s] for s in SENSORS])

        feats["total_dev_roll20"] = float(np.abs(devs_cur).mean())
        feats["total_dev_roll10"] = float(np.abs(devs_roll).mean())
        feats["obs_time_min"]     = float(t_vals[i])

        rows.append(feats)

    return rows


# ============================================================
# DATA AUGMENTATION + TARGET CREATION
# ============================================================
def augment_runs(df, stable_ref, stable_std, stable_time, n_runs=N_AUG_RUNS):
    """
    Generate n_runs augmented copies via time-warp + sensor noise.

    Target per row (1 + len(SENSORS) columns):
        [time_to_stability, sensor_1_at_future, ..., sensor_N_at_future]
    """
    np.random.seed(RANDOM_SEED)
    all_X, all_y, all_groups = [], [], []

    log(f"[3/6] Generating {n_runs} augmented runs...")

    for run_id in range(n_runs):
        speed       = np.random.uniform(*SPEED_RANGE)
        noise_scale = np.random.uniform(*NOISE_SCALE_RNG)

        aug = df.copy()
        aug["elapsed_time_min"] = aug["elapsed_time_min"] / speed

        for s in SENSORS:
            aug[s] = aug[s] + np.random.normal(0, stable_std[s] * noise_scale, len(aug))

        aug_stable = stable_time / speed
        rows       = engineer_features_batch(aug, stable_ref, stable_std)
        t_vals     = aug["elapsed_time_min"].values

        targets = []
        for i in range(20, len(aug)):
            time_target   = max(aug_stable - t_vals[i], 0.0)
            future_idx    = min(i + FUTURE_HORIZON, len(aug) - 1)
            sensor_future = [float(aug[s].iloc[future_idx]) for s in SENSORS]
            targets.append([time_target] + sensor_future)

        all_X.extend(rows)
        all_y.extend(targets)
        all_groups.extend([run_id] * len(rows))

        log(f"  Run {run_id+1:02d}/{n_runs} | Speed={speed:.2f} | "
            f"Noise={noise_scale:.2f} | Samples={len(rows)}")

    log("Augmentation complete")
    return pd.DataFrame(all_X), np.array(all_y), np.array(all_groups)


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================
def make_objective(Xs, y, groups):
    gkf = GroupKFold(n_splits=OPTUNA_CV_FOLDS)

    def objective(trial):
        params = {
            "max_iter":          trial.suggest_int("max_iter", 100, 300),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth":         trial.suggest_int("max_depth", 2, 5),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 10, 40),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 2.0),
            "max_bins":          trial.suggest_int("max_bins", 64, 255),
        }

        maes = []
        for tr, vl in gkf.split(Xs, y, groups):
            m = MultiOutputRegressor(
                HistGradientBoostingRegressor(**params, random_state=RANDOM_SEED)
            )
            m.fit(Xs[tr], y[tr])
            preds = m.predict(Xs[vl])
            # Optimise on time-to-stability only (output index 0)
            maes.append(mean_absolute_error(y[vl][:, 0], preds[:, 0]))

        return float(np.mean(maes))

    return objective


# ============================================================
# TRAINING PIPELINE
# ============================================================
def train(data_path, output_path="model.pkl"):
    start = time.time()

    log("=" * 60)
    log("Compressor Stability — Multi-Output Training Pipeline")
    log("=" * 60)

    # 1. Load data
    log("[1/6] Loading data...")
    df = pd.read_excel(data_path) if data_path.endswith(".xlsx") else pd.read_csv(data_path)
    df = df.sort_values("elapsed_time_min").reset_index(drop=True)
    log(f"Rows: {len(df)} | Phases: {df['phase'].value_counts().to_dict()}")

    # 2. Stable reference from Phase3
    log("[2/6] Computing stable reference (Phase3_StableRated)...")
    phase3 = df[df["phase"] == "Phase3_StableRated"]
    if phase3.empty:
        raise ValueError("No Phase3_StableRated rows found — check your data file.")

    STABLE_REF  = phase3[SENSORS].mean()
    STABLE_STD  = phase3[SENSORS].std().replace(0, 1e-6)
    STABLE_TIME = float(phase3["elapsed_time_min"].min())

    log(f"Stable phase onset: {STABLE_TIME:.2f} min")
    for s in SENSORS:
        log(f"  {s}: ref={STABLE_REF[s]:.3f}  std={STABLE_STD[s]:.4f}")

    # 3. Augment
    X, y, groups = augment_runs(df, STABLE_REF, STABLE_STD, STABLE_TIME, N_AUG_RUNS)
    FEATURES = list(X.columns)
    log(f"Feature matrix: {X.shape} | Target matrix: {y.shape}")

    # 4. Scale features only (targets are NOT scaled)
    log("[4/6] Scaling features...")
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    # 5. Optuna hyperparameter search
    log(f"[5/6] Running Optuna ({OPTUNA_TRIALS} trials, TPE sampler)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(make_objective(Xs, y, groups), n_trials=OPTUNA_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    log(f"Best CV MAE (time-to-stability): {study.best_value:.3f} min")
    log("Best params:")
    for k, v in best_params.items():
        log(f"  {k}: {v}")

    # 6. Train final model on full dataset
    log("[6/6] Training final model on full dataset...")
    final_model = MultiOutputRegressor(
        HistGradientBoostingRegressor(**best_params, random_state=RANDOM_SEED)
    )
    final_model.fit(Xs, y)
    log("Final model training complete")

    # Final cross-validation (evaluation only — does not replace final_model)
    log("Running final cross-validation for reporting...")
    gkf  = GroupKFold(n_splits=CV_FOLDS)
    maes = []
    for fold_num, (tr, vl) in enumerate(gkf.split(Xs, y, groups), start=1):
        m = MultiOutputRegressor(
            HistGradientBoostingRegressor(**best_params, random_state=RANDOM_SEED)
        )
        m.fit(Xs[tr], y[tr])
        preds = m.predict(Xs[vl])
        mae   = mean_absolute_error(y[vl][:, 0], preds[:, 0])
        maes.append(mae)
        log(f"  Fold {fold_num}/{CV_FOLDS} MAE: {mae:.3f} min")

    cv_mae_mean = float(np.mean(maes))
    cv_mae_std  = float(np.std(maes))
    log(f"Final CV MAE: {cv_mae_mean:.2f} ± {cv_mae_std:.2f} min")

    # Save model bundle — keys match what CompressorInference expects
    bundle = {
        "model":          final_model,   # MultiOutputRegressor
        "scaler":         scaler,
        "features":       FEATURES,
        "sensors":        SENSORS,
        "future_horizon": FUTURE_HORIZON,
        "stable_ref":     STABLE_REF,
        "stable_std":     STABLE_STD,
        "stable_time":    STABLE_TIME,
        "best_params":    best_params,
        "cv_mae":         cv_mae_mean,
        "cv_mae_std":     cv_mae_std,
    }

    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

    elapsed_min = (time.time() - start) / 60
    log("=" * 60)
    log(f"Training complete in {elapsed_min:.2f} minutes")
    log(f"CV MAE: {cv_mae_mean:.2f} ± {cv_mae_std:.2f} min")
    log(f"Model saved to: {output_path}")
    log("=" * 60)

    return bundle

### MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default=str(_DEFAULT_TRAIN_DATA))
    parser.add_argument("--output", default=str(_DEFAULT_MODEL_OUT))

    args = parser.parse_args()

    train(args.data, args.output)


# # ============================================================
# # MAIN
# # ============================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train compressor stability model")
#     parser.add_argument(
#         "--data",
#         default="KES22_8p5_synthetic_test_data.xlsx",
#         help="Path to test data file (.xlsx or .csv)",
#     )
#     parser.add_argument(
#         "--output",
#         default="model.pkl",
#         help="Output path for model bundle (.pkl)",
#     )
#     args = parser.parse_args()
#     train(args.data, args.output)