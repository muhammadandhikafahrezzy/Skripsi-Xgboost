from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PYTHON_TAG = f"{sys.version_info.major}{sys.version_info.minor}"
VERSIONED_DEPS = PROJECT_DIR / f".ml_deps_py{PYTHON_TAG}"
LOCAL_DEPS = PROJECT_DIR / ".ml_deps"
MPL_CACHE_DIR = PROJECT_DIR / ".matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

# Dependencies that contain compiled packages, such as scipy/xgboost/lightgbm,
# are specific to the Python minor version that installed them.
if VERSIONED_DEPS.exists():
    sys.path.insert(0, str(VERSIONED_DEPS))
elif sys.version_info[:2] != (3, 12) and LOCAL_DEPS.exists():
    raise SystemExit(
        "\n".join(
            [
                f"Script dijalankan dengan Python {sys.version_info.major}.{sys.version_info.minor}.",
                "Folder dependency `.ml_deps` yang tersedia dibuat untuk Python 3.12,",
                "jadi scipy/xgboost/lightgbm tidak kompatibel dengan Python ini.",
                "",
                "Solusi paling mudah:",
                r"  & 'C:\Users\DaysD\OneDrive\Documents\New project\run_model_4kategori.bat'",
                "",
                "Atau ubah interpreter VS Code ke Python 3.12 yang dipakai runner tersebut.",
            ]
        )
    )

import numpy as np
import pandas as pd

# For the bundled Python 3.12 runtime, import pandas/numpy first from the
# runtime, then expose locally installed ML packages.
if not VERSIONED_DEPS.exists() and LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

from joblib import dump
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_CATEGORIES = ["AIR MINERAL", "MINUMAN TEH", "ROKOK", "SUSU"]
RANDOM_STATE = 42
LAGS = [1, 2, 3, 7, 14, 28]
ROLLING_WINDOWS = [3, 7, 14]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def sanitize_name(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
    )


def load_daily_data(input_path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    required = {"TANGGAL", "KATEGORI", "QTY"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")

    df = df.copy()
    df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], errors="coerce").dt.normalize()
    df["QTY"] = pd.to_numeric(df["QTY"], errors="coerce")
    df = df.dropna(subset=["TANGGAL", "KATEGORI", "QTY"])
    df["KATEGORI"] = df["KATEGORI"].astype(str).str.strip().str.upper()
    df = df[df["KATEGORI"].isin(TARGET_CATEGORIES)]

    if df.empty:
        raise ValueError("Tidak ada baris yang cocok dengan 4 kategori target.")

    daily = (
        df.groupby(["TANGGAL", "KATEGORI"], as_index=False)["QTY"]
        .sum()
        .sort_values(["TANGGAL", "KATEGORI"])
    )

    # The transaction file has many calendar dates with no records at all.
    # To avoid treating non-recorded dates as certain zero demand, use recorded
    # business dates and fill only missing category rows inside those dates.
    recorded_dates = pd.Index(sorted(daily["TANGGAL"].unique()))
    full_index = pd.MultiIndex.from_product(
        [recorded_dates, TARGET_CATEGORIES], names=["TANGGAL", "KATEGORI"]
    )
    daily = (
        daily.set_index(["TANGGAL", "KATEGORI"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values(["KATEGORI", "TANGGAL"])
        .reset_index(drop=True)
    )
    return daily


def add_time_features(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily.copy()
    data["YEAR"] = data["TANGGAL"].dt.year
    data["MONTH"] = data["TANGGAL"].dt.month
    data["DAY"] = data["TANGGAL"].dt.day
    data["DAYOFWEEK"] = data["TANGGAL"].dt.dayofweek
    data["DAYOFYEAR"] = data["TANGGAL"].dt.dayofyear
    data["WEEKOFYEAR"] = data["TANGGAL"].dt.isocalendar().week.astype(int)
    data["QUARTER"] = data["TANGGAL"].dt.quarter
    data["IS_MONTH_START"] = data["TANGGAL"].dt.is_month_start.astype(int)
    data["IS_MONTH_END"] = data["TANGGAL"].dt.is_month_end.astype(int)
    data["MONTH_SIN"] = np.sin(2 * np.pi * data["MONTH"] / 12)
    data["MONTH_COS"] = np.cos(2 * np.pi * data["MONTH"] / 12)
    data["DOW_SIN"] = np.sin(2 * np.pi * data["DAYOFWEEK"] / 7)
    data["DOW_COS"] = np.cos(2 * np.pi * data["DAYOFWEEK"] / 7)

    grouped = data.groupby("KATEGORI", group_keys=False)["QTY"]
    for lag in LAGS:
        data[f"LAG_{lag}"] = grouped.shift(lag)

    shifted = data.groupby("KATEGORI", group_keys=False)["QTY"].shift(1)
    for window in ROLLING_WINDOWS:
        data[f"ROLLING_MEAN_{window}"] = shifted.groupby(data["KATEGORI"]).transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        data[f"ROLLING_STD_{window}"] = shifted.groupby(data["KATEGORI"]).transform(
            lambda s: s.rolling(window=window, min_periods=2).std()
        )

    data["EXPANDING_MEAN"] = shifted.groupby(data["KATEGORI"]).transform(
        lambda s: s.expanding(min_periods=2).mean()
    )
    data["EXPANDING_STD"] = shifted.groupby(data["KATEGORI"]).transform(
        lambda s: s.expanding(min_periods=3).std()
    )

    data = pd.get_dummies(data, columns=["KATEGORI"], prefix="KATEGORI", dtype=int)
    data = data.dropna().reset_index(drop=True)
    return data


def split_by_date(data: pd.DataFrame, test_ratio: float, val_ratio: float):
    dates = np.array(sorted(data["TANGGAL"].unique()))
    if len(dates) < 50:
        raise ValueError("Jumlah tanggal terlalu sedikit untuk split time series.")

    test_count = max(1, int(math.ceil(len(dates) * test_ratio)))
    train_val_dates = dates[:-test_count]
    test_dates = dates[-test_count:]

    val_count = max(1, int(math.ceil(len(train_val_dates) * val_ratio)))
    train_dates = train_val_dates[:-val_count]
    val_dates = train_val_dates[-val_count:]

    train = data[data["TANGGAL"].isin(train_dates)].copy()
    val = data[data["TANGGAL"].isin(val_dates)].copy()
    train_val = data[data["TANGGAL"].isin(train_val_dates)].copy()
    test = data[data["TANGGAL"].isin(test_dates)].copy()
    return train, val, train_val, test


def model_grids():
    xgb_base = {
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "n_jobs": 2,
        "tree_method": "hist",
    }
    xgb_grid = []
    for n_estimators, max_depth, learning_rate, reg_lambda in itertools.product(
        [200, 400], [3, 5], [0.03, 0.07], [1.0, 3.0]
    ):
        params = {
            **xgb_base,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": reg_lambda,
            "min_child_weight": 1.0,
        }
        xgb_grid.append(params)

    lgbm_base = {
        "objective": "regression",
        "random_state": RANDOM_STATE,
        "n_jobs": 2,
        "verbose": -1,
        "force_col_wise": True,
    }
    lgbm_grid = []
    for n_estimators, num_leaves, learning_rate, min_child_samples in itertools.product(
        [200, 400], [15, 31], [0.03, 0.07], [10, 20]
    ):
        params = {
            **lgbm_base,
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }
        lgbm_grid.append(params)

    return {
        "XGBoost": (XGBRegressor, xgb_grid),
        "LightGBM": (LGBMRegressor, lgbm_grid),
    }


def tune_and_fit(model_name, model_class, params_grid, x_train, y_train, x_val, y_val, x_train_val, y_train_val):
    best = None
    trials = []
    for params in params_grid:
        model = model_class(**params)
        model.fit(x_train, y_train)
        pred = np.maximum(model.predict(x_val), 0)
        score_rmse = rmse(y_val, pred)
        score_mae = mae(y_val, pred)
        score_mape = mape(y_val, pred)
        trials.append(
            {
                "ALGORITMA": model_name,
                "VALIDATION_RMSE": score_rmse,
                "VALIDATION_MAE": score_mae,
                "VALIDATION_MAPE": score_mape,
                "PARAMS": json.dumps(params, sort_keys=True),
            }
        )
        if best is None or score_rmse < best["rmse"]:
            best = {"rmse": score_rmse, "mae": score_mae, "mape": score_mape, "params": params}

    final_model = model_class(**best["params"])
    final_model.fit(x_train_val, y_train_val)
    return final_model, best, pd.DataFrame(trials).sort_values("VALIDATION_RMSE")


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algorithm, category), part in predictions.groupby(["ALGORITMA", "KATEGORI"]):
        rows.append(
            {
                "ALGORITMA": algorithm,
                "KATEGORI": category,
                "N_OBSERVASI_TEST": len(part),
                "RMSE": rmse(part["QTY"], part["PREDIKSI_QTY"]),
                "MAE": mae(part["QTY"], part["PREDIKSI_QTY"]),
                "MAPE": mape(part["QTY"], part["PREDIKSI_QTY"]),
            }
        )
    for algorithm, part in predictions.groupby("ALGORITMA"):
        rows.append(
            {
                "ALGORITMA": algorithm,
                "KATEGORI": "SEMUA_KATEGORI",
                "N_OBSERVASI_TEST": len(part),
                "RMSE": rmse(part["QTY"], part["PREDIKSI_QTY"]),
                "MAE": mae(part["QTY"], part["PREDIKSI_QTY"]),
                "MAPE": mape(part["QTY"], part["PREDIKSI_QTY"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["ALGORITMA", "KATEGORI"])


def recover_category(row: pd.Series) -> str:
    for category in TARGET_CATEGORIES:
        col = f"KATEGORI_{category}"
        if col in row and row[col] == 1:
            return category
    return "UNKNOWN"


def make_plots(predictions: pd.DataFrame, output_dir: Path) -> list[Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []
    for category in TARGET_CATEGORIES:
        part = predictions[predictions["KATEGORI"] == category].copy()
        if part.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        actual = part.drop_duplicates("TANGGAL").sort_values("TANGGAL")
        ax.plot(actual["TANGGAL"], actual["QTY"], label="Aktual", color="#111827", linewidth=2)
        for algorithm, alg_part in part.groupby("ALGORITMA"):
            alg_part = alg_part.sort_values("TANGGAL")
            ax.plot(alg_part["TANGGAL"], alg_part["PREDIKSI_QTY"], label=algorithm, linewidth=1.7)
        ax.set_title(f"Aktual vs Prediksi - {category}")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("QTY")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        out = plot_dir / f"actual_vs_prediksi_{sanitize_name(category)}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        plot_paths.append(out)
    return plot_paths


def write_summary(
    output_dir: Path,
    input_path: Path,
    daily: pd.DataFrame,
    feature_data: pd.DataFrame,
    metrics: pd.DataFrame,
    best_params: dict,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
):
    best_rows = metrics[metrics["KATEGORI"] == "SEMUA_KATEGORI"].sort_values("RMSE")
    best_algorithm = best_rows.iloc[0]["ALGORITMA"]
    metrics_table = metrics.copy()
    for col in ["RMSE", "MAE", "MAPE"]:
        metrics_table[col] = metrics_table[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    metric_headers = list(metrics_table.columns)
    metric_rows = metrics_table.astype(str).values.tolist()
    markdown_table = [
        "| " + " | ".join(metric_headers) + " |",
        "| " + " | ".join(["---"] * len(metric_headers)) + " |",
    ]
    markdown_table.extend("| " + " | ".join(row) + " |" for row in metric_rows)

    text = [
        "# Ringkasan Model Peramalan 4 Kategori",
        "",
        f"Sumber data: `{input_path}`",
        f"Kategori: {', '.join(TARGET_CATEGORIES)}",
        f"Rentang tanggal data harian: {daily['TANGGAL'].min().date()} sampai {daily['TANGGAL'].max().date()}",
        f"Jumlah baris harian sebelum fitur: {len(daily):,}",
        f"Jumlah baris setelah fitur lag/rolling: {len(feature_data):,}",
        "",
        "## Split Data",
        "",
        f"- Train: {train['TANGGAL'].min().date()} sampai {train['TANGGAL'].max().date()} ({train['TANGGAL'].nunique()} tanggal)",
        f"- Validation: {val['TANGGAL'].min().date()} sampai {val['TANGGAL'].max().date()} ({val['TANGGAL'].nunique()} tanggal)",
        f"- Test: {test['TANGGAL'].min().date()} sampai {test['TANGGAL'].max().date()} ({test['TANGGAL'].nunique()} tanggal)",
        "",
        "## Evaluasi Test",
        "",
        "\n".join(markdown_table),
        "",
        f"Algoritma terbaik berdasarkan RMSE keseluruhan: **{best_algorithm}**.",
        "",
        "## Hyperparameter Terbaik",
        "",
    ]
    for algorithm, params in best_params.items():
        text.append(f"### {algorithm}")
        text.append("")
        text.append("```json")
        text.append(json.dumps(params, indent=2, sort_keys=True))
        text.append("```")
        text.append("")
    (output_dir / "ringkasan_model.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost and LightGBM demand forecasting models for 4 thesis categories."
    )
    parser.add_argument(
        "--input",
        default=r"C:\Users\DaysD\Downloads\LPJ_KOPKAR_2024_2025_FAKTUR_MIRIP.xlsx",
        help="Path file Excel sumber.",
    )
    parser.add_argument("--sheet", default="LPJ", help="Nama sheet transaksi.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "outputs" / "model_4kategori"),
        help="Folder output hasil training.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_DIR / "models"),
        help="Folder output file model joblib.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    daily = load_daily_data(input_path, args.sheet)
    feature_data = add_time_features(daily)

    category_cols = [f"KATEGORI_{category}" for category in TARGET_CATEGORIES]
    for col in category_cols:
        if col not in feature_data.columns:
            feature_data[col] = 0

    drop_cols = {"TANGGAL", "QTY"}
    feature_cols = [c for c in feature_data.columns if c not in drop_cols]
    feature_cols = [c for c in feature_cols if c != "KATEGORI"]

    train, val, train_val, test = split_by_date(feature_data, args.test_ratio, args.val_ratio)
    x_train, y_train = train[feature_cols], train["QTY"]
    x_val, y_val = val[feature_cols], val["QTY"]
    x_train_val, y_train_val = train_val[feature_cols], train_val["QTY"]
    x_test, y_test = test[feature_cols], test["QTY"]

    all_predictions = []
    all_trials = []
    best_params = {}
    feature_importance = []

    for model_name, (model_class, grid) in model_grids().items():
        model, best, trials = tune_and_fit(
            model_name,
            model_class,
            grid,
            x_train,
            y_train,
            x_val,
            y_val,
            x_train_val,
            y_train_val,
        )
        best_params[model_name] = best["params"]
        all_trials.append(trials)

        pred = np.maximum(model.predict(x_test), 0)
        pred_frame = test[["TANGGAL", "QTY"] + category_cols].copy()
        pred_frame["KATEGORI"] = pred_frame.apply(recover_category, axis=1)
        pred_frame["ALGORITMA"] = model_name
        pred_frame["PREDIKSI_QTY"] = pred
        pred_frame["PREDIKSI_QTY_BULAT"] = np.rint(pred).astype(int)
        pred_frame = pred_frame[
            ["ALGORITMA", "TANGGAL", "KATEGORI", "QTY", "PREDIKSI_QTY", "PREDIKSI_QTY_BULAT"]
        ]
        all_predictions.append(pred_frame)

        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame(
                {
                    "ALGORITMA": model_name,
                    "FITUR": feature_cols,
                    "IMPORTANCE": model.feature_importances_,
                }
            ).sort_values("IMPORTANCE", ascending=False)
            feature_importance.append(imp)

        model_payload = {
            "model": model,
            "feature_columns": feature_cols,
            "target_categories": TARGET_CATEGORIES,
            "lags": LAGS,
            "rolling_windows": ROLLING_WINDOWS,
            "best_params": best["params"],
            "training_input": str(input_path),
            "sheet_name": args.sheet,
        }
        dump(model_payload, model_dir / f"{sanitize_name(model_name)}_4kategori.joblib")

    predictions = pd.concat(all_predictions, ignore_index=True).sort_values(
        ["ALGORITMA", "KATEGORI", "TANGGAL"]
    )
    metrics = evaluate_predictions(predictions)
    trials = pd.concat(all_trials, ignore_index=True)

    daily.to_csv(output_dir / "dataset_harian_4kategori.csv", index=False)
    feature_data.to_csv(output_dir / "dataset_fitur_4kategori.csv", index=False)
    predictions.to_csv(output_dir / "prediksi_test_4kategori.csv", index=False)
    metrics.to_csv(output_dir / "evaluasi_model_4kategori.csv", index=False)
    trials.to_csv(output_dir / "hasil_tuning_validation.csv", index=False)

    if feature_importance:
        pd.concat(feature_importance, ignore_index=True).to_csv(
            output_dir / "feature_importance.csv", index=False
        )

    plot_paths = make_plots(predictions, output_dir)
    write_summary(
        output_dir=output_dir,
        input_path=input_path,
        daily=daily,
        feature_data=feature_data,
        metrics=metrics,
        best_params=best_params,
        train=train,
        val=val,
        test=test,
    )

    print("Training selesai.")
    print(f"Output dir: {output_dir}")
    print(f"Model dir: {model_dir}")
    print(metrics.to_string(index=False))
    print("Plot files:")
    for path in plot_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
