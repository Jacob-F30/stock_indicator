from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

RequiredColumns = Literal[
    "Date",
    "Close",
    "Open",
    "High",
    "Low",
    "Volume",
    "PE_Ratio",
    "ROIC",
    "ROI",
    "Market_Cap",
]

CLASS_TO_ACTION: Dict[int, str] = {
    1: "BUY",
    0: "HOLD",
    -1: "SELL",
}


@dataclass
class ModelArtifacts:
    """Container for fitted model artifacts."""

    model: XGBClassifier
    feature_columns: list[str]
    classes_: list[int]


def _validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _compute_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess historical stock data and engineer indicators.

    Args:
        raw_df: Historical stock data with EOD prices and fundamentals.

    Returns:
        A cleaned DataFrame sorted by Date with technical/fundamental features.

    Raises:
        ValueError: If required columns are missing or not enough rows exist.
        TypeError: If input is not a DataFrame.
    """
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError("raw_df must be a pandas DataFrame")

    required: tuple[RequiredColumns, ...] = (
        "Date",
        "Close",
        "Open",
        "High",
        "Low",
        "Volume",
        "PE_Ratio",
        "ROIC",
        "ROI",
        "Market_Cap",
    )
    _validate_columns(raw_df, required)

    if len(raw_df) < 60:
        raise ValueError("At least 60 rows are required for stable feature generation")

    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    numeric_cols = [
        "Close",
        "Open",
        "High",
        "Low",
        "Volume",
        "PE_Ratio",
        "ROIC",
        "ROI",
        "Market_Cap",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    if df[numeric_cols].isna().any().any():
        raise ValueError("Unable to impute missing numeric values from input data")

    df["Daily_Return"] = df["Close"].pct_change().fillna(0.0)
    df["Volatility_EMA20"] = (
        df["Daily_Return"].abs().ewm(span=20, adjust=False, min_periods=20).mean()
    )
    df["Volatility_EMA20"] = df["Volatility_EMA20"].fillna(
        df["Volatility_EMA20"].median(skipna=True)
    )
    if df["Volatility_EMA20"].isna().any():
        df["Volatility_EMA20"] = df["Volatility_EMA20"].fillna(0.0)

    df["RSI_14"] = _compute_rsi(df["Close"], period=14)
    macd, macd_signal, macd_hist = _compute_macd(df["Close"])
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd_hist

    return df


def triple_barrier_label(
    df: pd.DataFrame,
    horizon_days: int = 20,
    upper_mult: float = 2.0,
    lower_mult: float = 1.0,
) -> pd.Series:
    """Generate Triple Barrier labels (+1, 0, -1).

    Args:
        df: Preprocessed DataFrame with Close and Volatility_EMA20.
        horizon_days: Vertical barrier in trading days.
        upper_mult: Upper barrier multiplier.
        lower_mult: Lower barrier multiplier.

    Returns:
        Label series aligned with df index.

    Raises:
        ValueError: If required columns are missing or horizon is invalid.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    _validate_columns(df, ["Close", "High", "Low", "Volatility_EMA20"])

    closes = df["Close"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    vols = df["Volatility_EMA20"].to_numpy(dtype=float)

    n = len(df)
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        if not np.isfinite(closes[i]) or closes[i] <= 0:
            labels[i] = 0
            continue

        vol = max(vols[i], 0.0)
        upper = closes[i] + (vol * upper_mult)
        lower = closes[i] - (vol * lower_mult)

        end = min(i + horizon_days, n - 1)
        if end <= i:
            labels[i] = 0
            continue

        future_highs = highs[i + 1 : end + 1]
        future_lows = lows[i + 1 : end + 1]

        up_touch = np.where(future_highs >= upper)[0]
        down_touch = np.where(future_lows <= lower)[0]

        first_up = up_touch[0] if up_touch.size > 0 else None
        first_down = down_touch[0] if down_touch.size > 0 else None

        if first_up is not None and (first_down is None or first_up <= first_down):
            labels[i] = 1
        elif first_down is not None and (first_up is None or first_down < first_up):
            labels[i] = -1
        else:
            labels[i] = 0

    return pd.Series(labels, index=df.index, name="Label")


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "PE_Ratio",
        "ROIC",
        "ROI",
        "Market_Cap",
        "Daily_Return",
        "Volatility_EMA20",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
    ]
    _validate_columns(df, feature_columns)

    X = df[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    X = X.ffill().bfill().fillna(0.0)
    return X, feature_columns


class TripleBarrierXGBoostAnalyst:
    """Triple-Barrier + XGBoost backend for market direction classification."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.artifacts: Optional[ModelArtifacts] = None

    def fit(self, historical_df: pd.DataFrame) -> dict[str, float]:
        """Fit the Analyst model with time-series aware evaluation.

        Args:
            historical_df: Raw historical DataFrame.

        Returns:
            Dict of cross-validation metrics.

        Raises:
            RuntimeError: If no valid training folds are produced.
        """
        horizon_days = 20
        processed = preprocess_data(historical_df)
        labels = triple_barrier_label(processed, horizon_days=horizon_days)

        valid_rows = (
            processed.index[:-horizon_days] if len(processed) > horizon_days else processed.index[:0]
        )
        if len(valid_rows) < 100:
            raise RuntimeError("Not enough labeled observations to train the model reliably")

        train_df = processed.loc[valid_rows].reset_index(drop=True)
        y = labels.loc[valid_rows].reset_index(drop=True)

        if y.nunique() < 2:
            raise RuntimeError("Training labels contain fewer than 2 classes")

        X, feature_columns = _build_feature_matrix(train_df)

        splitter = TimeSeriesSplit(n_splits=5)
        fold_scores: list[float] = []

        for train_idx, test_idx in splitter.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if y_train.nunique() < 2:
                continue

            fold_classes = sorted(int(cls) for cls in y_train.unique().tolist())
            class_to_idx = {cls: idx for idx, cls in enumerate(fold_classes)}
            idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
            y_train_enc = y_train.map(class_to_idx).astype(int)

            eval_mask = y_test.isin(fold_classes)
            if not eval_mask.any():
                continue
            y_test_eval = y_test[eval_mask]
            X_test_eval = X_test.loc[eval_mask]

            model = XGBClassifier(
                objective="multi:softprob",
                num_class=len(fold_classes),
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=self.random_state,
                eval_metric="mlogloss",
                n_jobs=2,
            )

            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_enc)
            model.fit(X_train, y_train_enc, sample_weight=sample_weights)
            pred_probs = model.predict_proba(X_test_eval)
            pred_idx = np.argmax(pred_probs, axis=1)
            pred_labels = pd.Series(pred_idx, index=y_test_eval.index).map(idx_to_class)
            fold_scores.append(f1_score(y_test_eval, pred_labels, average="macro"))

        if not fold_scores:
            raise RuntimeError("No valid folds generated during TimeSeriesSplit training")

        final_classes = sorted(int(cls) for cls in y.unique().tolist())
        class_to_idx = {cls: idx for idx, cls in enumerate(final_classes)}
        y_enc = y.map(class_to_idx).astype(int)

        final_model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(final_classes),
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=self.random_state,
            eval_metric="mlogloss",
            n_jobs=2,
        )
        final_weights = compute_sample_weight(class_weight="balanced", y=y_enc)
        final_model.fit(X, y_enc, sample_weight=final_weights)

        self.artifacts = ModelArtifacts(
            model=final_model,
            feature_columns=feature_columns,
            classes_=final_classes,
        )

        return {
            "cv_f1_macro_mean": float(np.mean(fold_scores)),
            "cv_f1_macro_std": float(np.std(fold_scores)),
            "training_rows": float(len(X)),
        }

    def predict_current(self, current_row: pd.DataFrame) -> dict[str, object]:
        """Predict class probabilities and recommendation for the latest row.

        Args:
            current_row: Single-row DataFrame already containing engineered features.

        Returns:
            Dict with class probabilities and recommended action.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If the row shape is invalid.
        """
        if self.artifacts is None:
            raise RuntimeError("Model is not fitted. Call fit(...) before predict_current(...)")

        if not isinstance(current_row, pd.DataFrame) or len(current_row) != 1:
            raise ValueError("current_row must be a single-row pandas DataFrame")

        feature_data = current_row.reindex(columns=self.artifacts.feature_columns)
        if feature_data.isna().all(axis=None):
            raise ValueError("current_row does not contain usable feature values")

        feature_data = feature_data.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        probabilities = self.artifacts.model.predict_proba(feature_data)[0]

        prob_map = {-1: 0.0, 0: 0.0, 1: 0.0}
        for idx, class_label in enumerate(self.artifacts.classes_):
            prob_map[class_label] = float(probabilities[idx])
        best_class = max(prob_map, key=lambda cls: prob_map[cls])

        return {
            "probabilities": {
                "+1": prob_map[1],
                "0": prob_map[0],
                "-1": prob_map[-1],
            },
            "recommended_action": CLASS_TO_ACTION[best_class],
            "predicted_class": int(best_class),
        }


def train_analyst_model(historical_df: pd.DataFrame) -> tuple[TripleBarrierXGBoostAnalyst, dict[str, float], pd.DataFrame]:
    """Convenience function to preprocess, train, and return a model with metrics."""
    analyst = TripleBarrierXGBoostAnalyst()
    metrics = analyst.fit(historical_df)
    processed = preprocess_data(historical_df)
    return analyst, metrics, processed
