# What Drives the Price of a Used Car?

A regression analysis of ~426,000 used vehicle listings (a Kaggle subset of a larger Craigslist dataset) aimed at identifying the factors that most influence used car sale prices. The work follows the CRISP-DM framework and is intended to produce actionable inventory guidance for a used car dealership.

**[Full analysis notebook → prompt_II.ipynb](prompt_II.ipynb)**

<mark>**Note to the grader:**</mark> **[prompt_II.ipynb](prompt_II.ipynb)** is the version of the assignment that I was able to complete before the due date of May 6th, the **[V2](V2)** directory is a version of the assignment that I continued working on after the due date just for my own curiosity.

## Approach

- **Cleaning:** dropped sentinel/placeholder values (year=1900, odometer=0 or 10M+, $0 prices), clipped extreme outliers (price > 99th percentile, odometer > 99.5th percentile), removed pre-1980 vehicles to focus on the daily-use market, dropped low-signal columns (`VIN`, `size`, `id`, `region`).
- **Feature engineering:** engineered `age = 2026 − year`; one-hot encoded low-cardinality categoricals; target-encoded `model` (~30k unique values) using sklearn's cross-fitted `TargetEncoder` to avoid leakage.
- **Modeling:** trained Linear Regression, Ridge, Lasso, and a Ridge model with polynomial features on the numeric columns. All models used `log(price)` as the target.
- **Validation:** 80/20 train/test split with `RidgeCV` / `LassoCV` cross-validation for hyperparameter selection.

## Key Findings

### Model performance

| Model                           | RMSE       | MAE        | R²        |
| ------------------------------- | ---------- | ---------- | --------- |
| Baseline (median)               | $13,455    | $10,672    | -0.04     |
| Linear Regression               | $8,497     | $5,485     | 0.584     |
| Ridge                           | $8,473     | $5,479     | 0.586     |
| Lasso                           | $8,453     | $5,458     | 0.588     |
| **Ridge + Polynomial Features** | **$7,614** | **$5,128** | **0.666** |

The polynomial Ridge model with added squared and interaction terms on `odometer`, `age`, and the target-encoded `model` produced the most accurate model, explaining ~67% of price variance with a median error of roughly $5,100. The improvement over plain Ridge confirms that depreciation curves are non-linear: each additional year of age and 10,000 miles of wear costs more on a near-new car than on an older one.

### Top drivers of price

**Push price up:**

- Specific high-value make/model (target-encoded `model` is the single strongest feature)
- Newer vehicles (steepest depreciation in the first ~5 years)
- Lower mileage (especially below 100k)
- `excellent`, `like new`, and `new` condition
- Pickups, trucks, coupes, and convertibles
- 8+ cylinder engines
- Diesel fuel
- Clean title status

**Push price down:**

- Salvage, rebuilt, or "parts only" titles (~30–50% discount)
- High mileage, particularly above 200k
- Vehicle age, especially past 10 years
- `fair` or `salvage` condition
- 3- or 4-cylinder engines
- Listings with multiple missing fields (a quality signal in itself)

### Recommendations for Inventory Decisions

1. **Prioritize clean-title inventory.** Title status is the single largest controllable risk factor in sourcing decisions.
2. **Target 0–10-year, sub-150k-mile vehicles** The most predictable pricing range.
3. **Trucks and pickups are higher-margin opportunities** in the used market. They hold value better than sedans of comparable age and mileage.
4. **Cosmetic features are less important** factors like paint color are not as important as they may seem. mechanical and structural features dominate price effects.
5. **Be cautious with sparse listings.** Vehicles with multiple missing fields tend to sell at a discount even when visible features look good.

## Limitations

- The model explains ~67% of price variation; the remaining ~33% lives in factors not captured here (listing photos, exact trim level, free-text descriptions, market timing, regional supply/demand).
- Most accurate on mid-priced vehicles ($10k–$30k); accuracy degrades at the high and low ends.
- Suitable for inventory-level strategy decisions, not as a per-vehicle pricing oracle.
- Pre-1980 vehicles were excluded; classic-car pricing follows different dynamics.
- Source data is Craigslist listings, dealer-network pricing patterns may differ.

## Suggested Next Steps

- Incorporate **listing-quality signals** (photo presence, description length, listing recency) as features.
- Use clustering to produce multiple pricing models for different car classifications.

## Repository Contents

- [`prompt_II.ipynb`](prompt_II.ipynb) — full CRISP-DM analysis notebook
- `data/` — raw input dataset
- `images/` — supporting images for the notebook
