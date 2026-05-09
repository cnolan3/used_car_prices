# What Drives the Price of a Used Car?

A regression analysis of ~426,000 used vehicle listings (a Kaggle subset of a larger Craigslist dataset) aimed at identifying the factors that most influence used car sale prices. The work follows the CRISP-DM framework and is intended to produce actionable inventory guidance for a used car dealership.

**[Analysis notebook → notebooks/01_analysis.ipynb](notebooks/01_analysis.ipynb)**
**[Dealer-facing report → notebooks/02_dealer_report.ipynb](notebooks/02_dealer_report.ipynb)**

## Approach

- **Cleaning:** dropped sentinel/placeholder values (year=1900, odometer=0 or 10M+, $0 prices), clipped extreme outliers (price > 99th percentile, odometer > 99.5th percentile), removed pre-1980 vehicles to focus on the daily-use market, dropped low-signal columns (`VIN`, `size`, `id`, `region`).
- **Feature engineering:** engineered `age = 2026 − year`; one-hot encoded low-cardinality categoricals; target-encoded `model` (~30k unique values) using sklearn's cross-fitted `TargetEncoder` to avoid leakage.
- **Modeling:** trained Linear Regression, Ridge, Lasso, and a Ridge model with polynomial features on the numeric columns. All models used `log(price)` as the target.
- **Segmentation:** applied PCA + KMeans clustering (K=7) to identify market segments, then fit a separate polynomial Ridge model per segment to capture per-segment pricing dynamics.
- **Validation:** 80/20 train/test split with `RidgeCV` / `LassoCV` cross-validation for hyperparameter selection.

## Key Findings

### Model performance

| Model                                | RMSE       | MAE        | R²        |
| ------------------------------------ | ---------- | ---------- | --------- |
| Baseline (median)                    | $13,455    | $10,672    | -0.04     |
| Linear Regression                    | $8,497     | $5,485     | 0.584     |
| Ridge                                | $8,473     | $5,479     | 0.586     |
| Lasso                                | $8,453     | $5,458     | 0.588     |
| Ridge + Polynomial Features          | $7,614     | $5,128     | 0.666     |
| **Per-cluster Polynomial Ridge**     | **$7,463** | **$4,728** | **0.679** |

The per-cluster polynomial Ridge produced the most accurate model, explaining ~68% of price variance with a median error of roughly $4,700. Two improvements drove the gains: (1) adding squared and interaction terms on `odometer`, `age`, and the target-encoded `model` confirmed that depreciation curves are non-linear — each additional year of age and 10,000 miles of wear costs more on a near-new car than on an older one — and (2) fitting separate polynomial Ridge models on each of seven market segments captured pricing dynamics that differ meaningfully between segments.

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

### Market segments

The clustering analysis identified seven distinct market segments, each with a different pricing profile. Two of them benefit substantially from segment-specific modeling, suggesting their pricing dynamics differ from the rest of the market:

| Segment                  | Share | Median price | Defining traits                                          |
| ------------------------ | ----- | ------------ | -------------------------------------------------------- |
| Economy sedans           | 21%   | $8,000       | 4-cyl, sedan, gas, Honda/Toyota, ~14 years / 103k miles  |
| HD diesel trucks         | 8%    | $33,000      | Diesel, V8, Ram/Ford HD, premium work trucks             |
| Mainstream SUVs          | 18%   | $15,000      | V6, Jeep/Ford/Honda, family-vehicle bread-and-butter     |
| Older V8 pickups         | 14%   | $14,000      | V8 gas, ~17 years old, Silverado/F-150 daily drivers     |
| Sparse listings          | 19%   | $10,500      | Multiple missing fields — a quality signal in itself     |
| Salvage/rebuilt **⭐**   | 3%    | $9,500       | Branded titles — discounted regardless of condition      |
| Newer premium/EV **⭐**  | 17%   | $29,500      | ~8 years / 25k miles, alt-fuel + premium pickups + EVs   |

Two segments deserve special attention because their pricing follows rules different from the broader market:

- **Salvage/rebuilt vehicles (3% of listings)** trade at a substantial discount even when other features are favorable. Title-branded inventory should be treated as a separate sourcing category with its own margin model.
- **Newer premium / alt-fuel / EV vehicles (17% of listings)** include modern luxury pickups, hybrids, and EVs. They follow a steeper depreciation curve than ICE vehicles and command higher absolute prices; segment-specific modeling reduced prediction error here by ~$2,800 per vehicle compared to the global model.

### Recommendations for Inventory Decisions

1. **Prioritize clean-title inventory.** Title status is the single largest controllable risk factor in sourcing decisions.
2. **Target 0–10-year, sub-150k-mile vehicles** for the most predictable pricing range and the bulk of market volume.
3. **Trucks and pickups are higher-margin opportunities** in the used market. They hold value better than sedans of comparable age and mileage.
4. **Cosmetic features are less important than they appear.** Paint color and similar cosmetic factors have weak effects compared to mechanical and structural attributes.
5. **Be cautious with sparse listings.** Vehicles with multiple missing fields tend to sell at a discount even when visible features look good.
6. **Treat salvage/rebuilt and newer alt-fuel/EV vehicles as separate inventory classes.** Both follow distinct pricing dynamics from the rest of the market and benefit from being sourced and priced under their own rules.

## Limitations

- The model explains ~68% of price variation; the remaining ~32% lives in factors not captured here (listing photos, exact trim level, free-text descriptions, market timing, regional supply/demand).
- Most accurate on mid-priced vehicles ($10k–$30k); accuracy degrades at the high and low ends.
- Suitable for inventory-level strategy decisions, not as a per-vehicle pricing oracle.
- Pre-1980 vehicles were excluded; classic-car pricing follows different dynamics.
- Source data is Craigslist listings, dealer-network pricing patterns may differ.
- Heavy-duty diesel trucks (Cluster 1) remain the hardest segment to price accurately, with per-cluster RMSE roughly 60% higher than the average — trim, towing capacity, and exact engine variant matter substantially within this segment but are not well captured in the dataset.

## Suggested Next Steps

- Incorporate **listing-quality signals** (photo presence, description length, listing recency) as features.
- Try a **tree-based model** (Random Forest or Gradient Boosting) on the same feature matrix — likely to push R² to ~0.80–0.85 by capturing non-linear effects across all features.
- Engineer **cross-feature interactions** (e.g., `manufacturer × age`, `type × odometer`) to capture brand- and category-specific depreciation patterns in the linear model.
- **Re-cluster with K=8 or K=9** to see whether the heavy-duty truck segment splits productively, since it currently has the worst per-cluster fit.

## Repository Contents

- [`notebooks/01_analysis.ipynb`](notebooks/01_analysis.ipynb) — full analysis notebook (data prep, modeling, clustering, evaluation)
- [`notebooks/02_dealer_report.ipynb`](notebooks/02_dealer_report.ipynb) — dealer-facing report (loads artifacts, presents findings)
- `src/utils.py` — shared helpers (`evaluate`, `make_poly_pipeline`, `CLUSTER_LABELS`)
- `data/` — raw input dataset
- `artifacts/` — generated outputs (trained models, predictions, metrics) used by the report notebook
