# Kaggle House Prices (Ames Housing)

Goal: Predict `SalePrice` using tabular regression.

## Baseline (v1)

- Model: Ridge (scikit-learn)
- Encoding: One-hot (`pd.get_dummies`)
- Missing handling: `fillna(0)`
- Alignment: `train/test align`
- Output: `outputs/submission_ridge_baseline.csv`

## Project Structure

```
data/
 └── raw/
src/
 └── train_ridge_baseline.py
outputs/
 └── submission_ridge_baseline.csv
```

## Reproduce

```bash
python src/train_ridge_baseline.py
```

## Next Steps

- Add LightGBM baseline
- Add cross-validation
- Feature engineering (total_sf, house_age, etc.)
- Track score improvements via commit messages
