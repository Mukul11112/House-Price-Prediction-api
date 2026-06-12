"""
preprocess.py
=============
Complete data preprocessing pipeline for House Price Prediction.

Steps performed (in order):
  1. Remove duplicate rows
  2. Impute missing values  (median for numeric, mode for categorical)
  3. Detect & cap outliers  (IQR / Tukey fences — winsorize, don't drop)
  4. Engineer new features  (house age, total area, bathroom count, flags)
  5. Encode categoricals    (ordinal for quality cols, OHE for nominals)
  6. Feature selection      (SelectKBest / F-regression on training data)
  7. Standard-scale numerics (StandardScaler fit on training data only)

The HouseDataPreprocessor class is stateful: fit on train, applied to test
and to single API requests, preventing any data leakage.
"""

import logging
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Ordinal encoding maps (Ames Housing domain knowledge) ────────────────────
ORDINAL_MAPS: Dict[str, Dict[str, int]] = {
    "ExterQual":   {"Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "KitchenQual": {"Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "HeatingQC":   {"Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
}

# ── Columns to cap outliers ──────────────────────────────────────────────────
OUTLIER_COLS = ["LotArea", "GrLivArea", "TotalBsmtSF", "GarageArea"]

# ── Nominal columns for one-hot encoding ────────────────────────────────────
OHE_COLS = ["MSZoning", "Neighborhood", "Foundation"]

# ── Base numeric feature list (before OHE expansion) ────────────────────────
NUMERIC_FEATURES = [
    "LotArea", "OverallQual", "YearBuilt", "YearRemodAdd",
    "TotalBsmtSF", "GrLivArea", "FullBath", "HalfBath",
    "BedroomAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
    "ExterQual", "KitchenQual", "HeatingQC",
    # engineered
    "HouseAge", "RemodelAge", "TotalSF", "TotalBathrooms",
    "HasGarage", "HasFireplace", "HasDeck",
]

TARGET = "SalePrice"


# ── Stateless helpers ────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Read CSV and log basic info."""
    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Removed {before - len(df)} duplicate row(s). Remaining: {len(df)}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute NaNs:
      • Numeric  → column median
      • Category → column mode
    Uses assignment instead of fillna(inplace=) for pandas-3 compatibility.
    """
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        n = df[col].isna().sum()
        if n:
            val = df[col].median()
            df[col] = df[col].fillna(val)
            logger.info(f"  Imputed {n} missing in '{col}' with median={val:.2f}")

    for col in df.select_dtypes(include=["object"]).columns:
        n = df[col].isna().sum()
        if n:
            val = df[col].mode()[0]
            df[col] = df[col].fillna(val)
            logger.info(f"  Imputed {n} missing in '{col}' with mode='{val}'")

    logger.info("Missing value imputation complete.")
    return df


def handle_outliers(df: pd.DataFrame, cols: List[str] = OUTLIER_COLS) -> pd.DataFrame:
    """
    Cap (winsorise) extreme values using Tukey's IQR fences:
        lower = Q1 - 1.5 × IQR
        upper = Q3 + 1.5 × IQR
    Rows are kept; only values are clipped.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lo, hi)
        logger.info(f"  Outlier cap '{col}': [{lo:.1f}, {hi:.1f}] — {n_clipped} clipped")
    return df


def engineer_features(df: pd.DataFrame, ref_year: int = 2010) -> pd.DataFrame:
    """
    Derive seven new features from existing columns:
      HouseAge      – age of house at reference year
      RemodelAge    – years since last remodel
      TotalSF       – combined above-ground + basement SF
      TotalBathrooms– full + 0.5 × half baths
      HasGarage     – 1 if garage exists
      HasFireplace  – 1 if at least one fireplace
      HasDeck       – 1 if wood deck present
    """
    df = df.copy()
    df["HouseAge"]       = ref_year - df["YearBuilt"]
    df["RemodelAge"]     = ref_year - df["YearRemodAdd"]
    df["TotalSF"]        = df["GrLivArea"] + df.get("TotalBsmtSF", pd.Series(0, index=df.index))
    df["TotalBathrooms"] = df["FullBath"] + 0.5 * df["HalfBath"]
    df["HasGarage"]      = (df["GarageCars"] > 0).astype(int)
    df["HasFireplace"]   = (df["Fireplaces"]  > 0).astype(int)
    df["HasDeck"]        = (df["WoodDeckSF"]  > 0).astype(int)
    logger.info("Feature engineering complete (7 new features).")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Ordinal-encode quality columns using domain-knowledge maps.
    2. One-hot encode nominal columns (drop_first=True avoids multicollinearity).
    """
    df = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(2).astype(int)
            logger.info(f"  Ordinal encoded '{col}'")

    for col in OHE_COLS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            logger.info(f"  One-hot encoded '{col}' → {dummies.shape[1]} cols")
    return df


def select_features(
    df: pd.DataFrame,
    target_col: str = TARGET,
    k: int = 20,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Keep the top-k features ranked by univariate F-regression score.
    Call ONLY on training data; apply the returned feature list to test data.
    """
    ohe_present = [c for c in df.columns if any(c.startswith(p + "_") for p in OHE_COLS)]
    candidates  = [f for f in NUMERIC_FEATURES if f in df.columns] + ohe_present

    X = df[candidates].copy()
    y = df[target_col]

    # Final safety net: fill any residual NaN with column medians
    X = X.fillna(X.median())

    selector = SelectKBest(f_regression, k=min(k, len(candidates)))
    selector.fit(X, y)
    selected = [f for f, keep in zip(candidates, selector.get_support()) if keep]
    logger.info(f"Feature selection: kept {len(selected)} / {len(candidates)}")
    return df[selected + [target_col]], selected


# ── Stateful pipeline class ──────────────────────────────────────────────────

class HouseDataPreprocessor:
    """
    Stateful pipeline that fits all data-dependent transformers (scaler,
    feature selector, imputation medians) on training data and reuses them
    for test data and live API requests.

    Usage
    -----
        pre = HouseDataPreprocessor()
        X_train, y_train = pre.fit_transform(train_df)
        X_test,  y_test  = pre.transform(test_df)
        x_api            = pre.transform_single({"overall_qual": 7, ...})
    """

    def __init__(self, k_features: int = 20):
        self.k_features       = k_features
        self.scaler           = StandardScaler()
        self.selected_features: List[str] = []
        self.feature_medians: Dict[str, float] = {}
        self.is_fitted        = False

    # ── private ─────────────────────────────────────────────────────────────
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the stateless cleaning steps."""
        df = remove_duplicates(df)
        df = handle_missing_values(df)
        df = handle_outliers(df)
        df = engineer_features(df)
        df = encode_categoricals(df)
        return df

    # ── public ──────────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on training data and return (X_scaled, y)."""
        logger.info("=== Fitting preprocessor on training data ===")
        df = self._clean(df)
        df, self.selected_features = select_features(df, k=self.k_features)

        X_df = df[self.selected_features].fillna(df[self.selected_features].median())
        self.feature_medians = X_df.median().to_dict()

        X = self.scaler.fit_transform(X_df)
        y = df[TARGET].values
        self.is_fitted = True
        logger.info(f"Preprocessor fitted. X shape: {X.shape}")
        return X, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform test / validation data."""
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform first.")
        df = self._clean(df)
        for col in self.selected_features:          # align OHE columns
            if col not in df.columns:
                df[col] = 0
        X_df = df[self.selected_features].fillna(pd.Series(self.feature_medians))
        X    = self.scaler.transform(X_df)
        y    = df[TARGET].values if TARGET in df.columns else np.array([])
        return X, y

    def transform_single(self, inp: Dict[str, Any]) -> np.ndarray:
        """
        Convert an API request dict to a scaled 2-D numpy array ready for
        model.predict(). Missing fields default to training-set medians.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform first.")

        # API key → internal feature name
        api_to_feat = {
            "overall_qual":  "OverallQual",
            "gr_liv_area":   "GrLivArea",
            "garage_cars":   "GarageCars",
            "total_bsmt_sf": "TotalBsmtSF",
            "full_bath":     "FullBath",
            "year_built":    "YearBuilt",
        }

        row = dict(self.feature_medians)   # start from medians
        for api_key, feat in api_to_feat.items():
            if api_key in inp and feat in row:
                row[feat] = float(inp[api_key])

        # Recompute engineered features from whatever was supplied
        yb               = float(inp.get("year_built", 1990))
        row["HouseAge"]      = 2010 - yb
        row["RemodelAge"]    = 2010 - yb
        row["TotalSF"]       = row.get("GrLivArea", 1500) + row.get("TotalBsmtSF", 1000)
        row["TotalBathrooms"]= row.get("FullBath", 2) + 0.5 * row.get("HalfBath", 0)
        row["HasGarage"]     = int(row.get("GarageCars", 2) > 0)
        row["HasFireplace"]  = int(row.get("Fireplaces",  0) > 0)
        row["HasDeck"]       = int(row.get("WoodDeckSF",  0) > 0)

        X_df = pd.DataFrame([row])[self.selected_features]
        return self.scaler.transform(X_df)
