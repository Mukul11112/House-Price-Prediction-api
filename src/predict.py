"""
predict.py
==========
Inference helpers used by the FastAPI app.

  load_model_artifact()  – load the joblib artifact from disk (cached)
  predict_price()        – run a single prediction from an API request dict
  predict_batch()        – run predictions on a list of request dicts
"""

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List

import joblib
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "house_price_model.pkl")


@lru_cache(maxsize=1)
def load_model_artifact() -> Dict[str, Any]:
    """
    Load the model artifact from disk exactly once (LRU-cached).

    Returns a dict with keys:
        model, preprocessor, model_name, selected_features,
        metrics, model_version, all_results
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Run `python src/train.py` first."
        )
    logger.info(f"Loading model artifact from: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    logger.info(
        f"Loaded model: {artifact['model_name']}  "
        f"v{artifact['model_version']}  "
        f"R²={artifact['metrics']['R2']:.4f}"
    )
    return artifact


def predict_price(input_dict: Dict[str, Any]) -> float:
    """
    Given a dict of house features (matching the POST /predict schema),
    return the predicted sale price as a Python float.
    """
    artifact     = load_model_artifact()
    model        = artifact["model"]
    preprocessor = artifact["preprocessor"]

    X = preprocessor.transform_single(input_dict)
    price = float(model.predict(X)[0])

    # Clip to a sensible range to guard against extreme extrapolation
    price = max(10_000.0, min(price, 2_000_000.0))
    logger.info(f"Prediction: ${price:,.2f}  input={input_dict}")
    return round(price, 2)


def predict_batch(inputs: List[Dict[str, Any]]) -> List[float]:
    """
    Run predictions on a list of feature dicts.
    Returns a list of predicted prices in the same order.
    """
    return [predict_price(inp) for inp in inputs]
