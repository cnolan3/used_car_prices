# src/utils.py
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeCV

CLUSTER_LABELS = { 
    0: "Economy sedans",
    1: "HD diesel trucks",
    2: "Mainstream SUVs",
    3: "Older V8 pickups",
    4: "Sparse listings",
    5: "Salvage/rebuilt",
    6: "Newer premium/EV",
}

NUMERIC_COLS = ['odometer', 'age', 'model']

def evaluate(name, y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{name:30s}  RMSE: ${rmse:>8,.0f}   MAE: ${mae:>8,.0f}   R²: {r2:.3f}")
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def make_poly_pipeline(all_columns, numeric_cols=NUMERIC_COLS, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    """Same structure as your global polynomial Ridge, but built fresh per cluster."""
    numeric_cols   = ['odometer', 'age', 'model']
    all_other_cols = [c for c in all_columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('poly',   PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
    ])

    # Drop any one-hot columns that are constant within this cluster
    # (e.g., a "trucks-only" cluster has manufacturer_ferrari=0 for every row,
    # which would break StandardScaler with a divide-by-zero)
    cat_pipe = Pipeline([
        ('drop_constant', VarianceThreshold(threshold=0.0)),
        ('scaler',         StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, numeric_cols),
        ('cat', cat_pipe,     all_other_cols),
    ])

    return Pipeline([
        ('preprocess', preprocessor),
        ('ridge',      RidgeCV(alphas=list(alphas))),
    ])