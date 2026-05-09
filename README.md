# What Drives the Price of a Used Car?

A regression analysis of ~426,000 used vehicle listings (a Kaggle subset of a larger Craigslist dataset) aimed at identifying the factors that most influence used car sale prices. The work follows the CRISP-DM framework and is intended to produce actionable inventory guidance for a used car dealership.

**[Analysis notebook → notebooks/01_analysis.ipynb](notebooks/01_analysis.ipynb)**
**[Dealer-facing report → notebooks/02_dealer_report.ipynb](notebooks/02_dealer_report.ipynb)**

## Approach

- **Cleaning:** dropped sentinel/placeholder values (year=1900, odometer=0 or 10M+, $0 prices), clipped extreme outliers (price > 99th percentile, odometer > 99.5th percentile), removed pre-1980 vehicles to focus on the daily-use market, dropped low-signal columns (`VIN`, `size`, `id`, `region`).
- **Feature engineering:** engineered `age = 2026 − year`; one-hot encoded low-cardinality categoricals; target-encoded `model` (~30k unique values) using sklearn's cross-fitted `TargetEncoder` to avoid leakage.
- **Modeling:** trained Linear Regression, Ridge, Lasso, and a Ridge model with polynomial features on the numeric columns. All models used `log(price)` as the target.
- **Segmentation:** applied PCA + KMeans clustering (K=8) to identify market segments, then fit a separate polynomial Ridge model per segment to capture per-segment pricing dynamics.
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
| **Per-cluster Polynomial Ridge**     | **$7,239** | **$4,603** | **0.698** |

The per-cluster polynomial Ridge produced the most accurate model, explaining ~70% of price variance with a median error of roughly $4,600. Two improvements drove the gains: (1) adding squared and interaction terms on `odometer`, `age`, and the target-encoded `model` confirmed that depreciation curves are non-linear — each additional year of age and 10,000 miles of wear costs more on a near-new car than on an older one — and (2) fitting separate polynomial Ridge models on each of eight market segments captured pricing dynamics that differ meaningfully between segments. Six of the eight clusters benefit from per-cluster modeling, with the gain concentrated in two distinct segments described below.

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

The clustering analysis identified eight distinct market segments, each with a different pricing profile. Two of them (marked ⭐) benefit substantially from segment-specific modeling, indicating their pricing dynamics differ meaningfully from the rest of the market:

| Segment                       | Share | Median price | Defining traits                                                |
| ----------------------------- | ----- | ------------ | -------------------------------------------------------------- |
| Mainstream SUVs               | 19%   | $9,700       | V6, Jeep / Ford / Honda; family vehicles                       |
| Older V8 gas pickups          | 14%   | $18,000      | F-150 / Silverado / Sierra daily-driver pickups                |
| HD diesel trucks              | 6%    | $32,500      | F-250/350 diesels, Ram HDs; premium work trucks                |
| **Newer premium pickups ⭐** | 11%   | $29,600      | ~8 years, ~24k miles; GMC Sierra / premium Silverados          |
| Economy sedans                | 13%   | $7,700       | 4-cyl, Honda/Toyota; high-volume cheap sedans                  |
| Mid-tier mixed                | 13%   | $16,700      | Heterogeneous mid-aged group; sparse condition data            |
| Sparse listings               | 16%   | $9,500       | Multiple missing fields — a quality signal in itself           |
| **Newer EV / alt-fuel ⭐**   | 6%    | $28,600      | EVs, hybrids, plug-ins; very low miles for the age             |

Two segments deserve special attention because their pricing follows rules different from the broader market:

- **Newer premium pickups (11% of listings)** are dominated by recent-vintage GMC Sierras, premium Silverados, and similar trim-heavy gas pickups with low mileage. Trim level and configuration drive significant price variation; segment-specific modeling reduced prediction error here by ~$1,700 per vehicle compared to the global model.
- **Newer EV / alt-fuel vehicles (6% of listings)** include modern EVs, plug-in hybrids, and alt-fuel vehicles. They follow a steeper depreciation curve than ICE vehicles and command higher absolute prices for their age and mileage; segment-specific modeling reduced prediction error here by ~$2,400 per vehicle compared to the global model — the largest improvement of any segment.

### Recommendations for Inventory Decisions

1. **Prioritize clean-title inventory.** Title status is the single largest controllable risk factor in sourcing decisions — salvage and rebuilt vehicles trade at roughly 30–50% discounts.
2. **Target 0–10-year, sub-150k-mile vehicles** for the most predictable pricing range and the bulk of market volume.
3. **Trucks and pickups are higher-margin opportunities** in the used market. They hold value better than sedans of comparable age and mileage.
4. **Cosmetic features are less important than they appear.** Paint color and similar cosmetic factors have weak effects compared to mechanical and structural attributes.
5. **Be cautious with sparse listings.** Vehicles with multiple missing fields tend to sell at a discount even when visible features look good.
6. **Treat the highlighted segments as separate inventory classes.** Newer premium pickups and newer EV/alt-fuel vehicles each follow distinct pricing dynamics from the rest of the market and benefit from being sourced and priced under their own rules rather than blanket "any 8-year-old vehicle" heuristics.

## Limitations

- The model explains ~70% of price variation; the remaining ~30% lives in factors not captured here (listing photos, exact trim level, free-text descriptions, market timing, regional supply/demand).
- Most accurate on mid-priced vehicles ($10k–$30k); accuracy degrades at the high and low ends.
- Suitable for inventory-level strategy decisions, not as a per-vehicle pricing oracle.
- Pre-1980 vehicles were excluded; classic-car pricing follows different dynamics.
- Source data is Craigslist listings, dealer-network pricing patterns may differ.
- Heavy-duty diesel trucks remain the hardest segment to price accurately, with per-cluster RMSE roughly 60% higher than the average — trim, towing capacity, and exact engine variant matter substantially within this segment but are not well captured in the dataset.

## Suggested Next Steps

- Incorporate **listing-quality signals** (photo presence, description length, listing recency) as features.
- Try a **tree-based model** (Random Forest or Gradient Boosting) on the same feature matrix — likely to push R² to ~0.80–0.85 by capturing non-linear effects across all features.
- Engineer **cross-feature interactions** (e.g., `manufacturer × age`, `type × odometer`) to capture brand- and category-specific depreciation patterns in the linear model.
- **Source additional features for the heavy-duty truck segment.** Trim level, engine variant, and tow-package configuration would deliver disproportionate accuracy gains for the hardest-to-fit cluster.

## Repository Contents

- [`notebooks/01_analysis.ipynb`](notebooks/01_analysis.ipynb) — full analysis notebook (data prep, modeling, clustering, evaluation)
- [`notebooks/02_dealer_report.ipynb`](notebooks/02_dealer_report.ipynb) — dealer-facing report (loads artifacts, presents findings)
- `src/utils.py` — shared helpers (`evaluate`, `make_poly_pipeline`, `CLUSTER_LABELS`)
- `data/` — raw input dataset
- `artifacts/` — generated outputs (trained models, predictions, metrics) used by the report notebook
