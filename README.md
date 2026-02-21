# Kaggle | House Prices: Advanced Regression Techniques

Baseline Pipeline: RobustScaler + Lasso  
Author: Haemin  

Goal: Predict `SalePrice` using tabular regression.

---

## Baseline (v2)

- Model: Lasso (scikit-learn)
- Numeric: Median Imputation + RobustScaler
- Categorical: Mode Imputation + OneHotEncoder
- Outlier handling: GrLivArea rule
- Validation: Holdout + 5-Fold CV
- Metric: RMSE
- CV Score: ~0.119
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
