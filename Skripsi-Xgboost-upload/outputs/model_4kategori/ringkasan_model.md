# Ringkasan Model Peramalan 4 Kategori

Sumber data: `C:\Users\DaysD\Downloads\LPJ_KOPKAR_2024_2025_FAKTUR_MIRIP.xlsx`
Kategori: AIR MINERAL, MINUMAN TEH, ROKOK, SUSU
Rentang tanggal data harian: 2024-01-03 sampai 2025-12-22
Jumlah baris harian sebelum fitur: 1,800
Jumlah baris setelah fitur lag/rolling: 1,688

## Split Data

- Train: 2024-02-10 sampai 2025-05-12 (269 tanggal)
- Validation: 2025-05-15 sampai 2025-08-24 (68 tanggal)
- Test: 2025-08-25 sampai 2025-12-22 (85 tanggal)

## Evaluasi Test

| ALGORITMA | KATEGORI | N_OBSERVASI_TEST | RMSE | MAE | MAPE |
| --- | --- | --- | --- | --- | --- |
| LightGBM | AIR MINERAL | 85 | 135.3573 | 109.4754 | 33.6198 |
| LightGBM | MINUMAN TEH | 85 | 47.6337 | 37.0892 | 25.0871 |
| LightGBM | ROKOK | 85 | 21.1584 | 16.7887 | 19.6075 |
| LightGBM | SEMUA_KATEGORI | 340 | 73.0096 | 43.9589 | 32.7126 |
| LightGBM | SUSU | 85 | 16.8330 | 12.4822 | 52.5360 |
| XGBoost | AIR MINERAL | 85 | 123.9649 | 102.9898 | 30.1719 |
| XGBoost | MINUMAN TEH | 85 | 46.9693 | 35.3936 | 22.5946 |
| XGBoost | ROKOK | 85 | 21.8296 | 17.2661 | 20.5912 |
| XGBoost | SEMUA_KATEGORI | 340 | 67.7914 | 42.5409 | 38.3532 |
| XGBoost | SUSU | 85 | 18.2416 | 14.5142 | 80.0551 |

Algoritma terbaik berdasarkan RMSE keseluruhan: **XGBoost**.

## Hyperparameter Terbaik

### XGBoost

```json
{
  "colsample_bytree": 0.9,
  "learning_rate": 0.03,
  "max_depth": 3,
  "min_child_weight": 1.0,
  "n_estimators": 200,
  "n_jobs": 2,
  "objective": "reg:squarederror",
  "random_state": 42,
  "reg_lambda": 1.0,
  "subsample": 0.9,
  "tree_method": "hist"
}
```

### LightGBM

```json
{
  "colsample_bytree": 0.9,
  "force_col_wise": true,
  "learning_rate": 0.03,
  "min_child_samples": 10,
  "n_estimators": 200,
  "n_jobs": 2,
  "num_leaves": 31,
  "objective": "regression",
  "random_state": 42,
  "subsample": 0.9,
  "verbose": -1
}
```
