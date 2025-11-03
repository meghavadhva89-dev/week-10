import pickle
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_1 = Path("model_1.pickle")
MODEL_2 = Path("model_2.pickle")
MODEL_3 = Path("model_3.pickle")


def predict_rating(df_X, text: bool = False):
    """Predict ratings for rows in ``df_X``.

    Args:
        df_X (pd.DataFrame): DataFrame with columns ``100g_USD`` (numeric)
            and ``roast`` (text).
        text (bool): If True, predict from text (uses text model).

    Returns:
        numpy.ndarray: Predicted ratings (float) for each row.

    Behavior:
        - If ``roast`` value is in ``model_2``'s ``roast_map``, use the
          DecisionTreeRegressor trained on (``100g_USD``, ``roast_cat``).
        - Otherwise fall back to ``model_1`` which uses only ``100g_USD``.
    """
    if text:
        # Support text prediction using model_3.pickle which contains
        # a dict with keys 'model' and 'vectorizer'
        # Accept either a DataFrame with a 'text' column or an array-like of strings
        texts = None
        if isinstance(df_X, pd.DataFrame):
            if "text" not in df_X.columns:
                raise ValueError(
                    (
                        "When text=True, df_X must be a DataFrame with a 'text' "
                        "column or an array-like of strings"
                    )
                )
            texts = df_X["text"].fillna("").astype(str).to_list()
        elif hasattr(df_X, "__iter__"):
            texts = [str(t) for t in df_X]
        else:
            raise ValueError(
                (
                    "Unsupported input for text=True; provide DataFrame with 'text' "
                    "column or array-like of strings"
                )
            )

        if not MODEL_3.exists():
            raise RuntimeError(
                (
                    "model_3.pickle not found. Train the text model first by "
                    "running train.py"
                )
            )

        with MODEL_3.open("rb") as f:
            obj = pickle.load(f)
        if not (isinstance(obj, dict) and "model" in obj and "vectorizer" in obj):
            raise RuntimeError(
                (
                    "model_3.pickle has unexpected format; expected dict with "
                    "'model' and 'vectorizer'"
                )
            )

        model3 = obj["model"]
        vectorizer = obj["vectorizer"]
        Xt = vectorizer.transform(texts)
        preds_text = model3.predict(Xt)
        return preds_text

    if not isinstance(df_X, pd.DataFrame):
        raise ValueError("df_X must be a pandas DataFrame")

    required = {"100g_USD", "roast"}
    if not required.issubset(set(df_X.columns)):
        raise ValueError(f"df_X must contain columns: {sorted(required)}")

    # Load models if available
    model2 = None
    roast_map = {}
    if MODEL_2.exists():
        with MODEL_2.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "model" in obj and "roast_map" in obj:
            model2 = obj["model"]
            roast_map = obj["roast_map"]

    model1 = None
    if MODEL_1.exists():
        with MODEL_1.open("rb") as f:
            model1 = pickle.load(f)

    if model2 is None and model1 is None:
        raise RuntimeError(
            (
                "No models found: expected model_1.pickle and/or model_2.pickle "
                "in repository"
            )
        )

    # Prepare input arrays
    X_price = df_X["100g_USD"].astype(float).to_numpy().reshape(-1, 1)
    roasts = df_X["roast"].fillna("").astype(str).to_numpy()

    preds = np.empty(len(df_X), dtype=float)
    preds[:] = np.nan

    # If model2 is available, map roast text -> roast_num using roast_map (unknown -> 0)
    if model2 is not None and roast_map:
        roast_nums = np.array([roast_map.get(r, 0) for r in roasts], dtype=float)
        X2 = np.column_stack([X_price.ravel(), roast_nums])
        use_model2 = roast_nums != 0
        if use_model2.any():
            preds_model2 = model2.predict(X2[use_model2])
            preds[use_model2] = preds_model2

    # For remaining rows or when model2 isn't available, use model1 (fallback)
    remaining = np.isnan(preds)
    if remaining.any():
        if model1 is None:
            # If model1 missing but model2 exists, predict using model2 with roast_num=0
            if model2 is not None:
                roast_nums = np.array(
                    [roast_map.get(r, 0) for r in roasts], dtype=float
                )
                X2 = np.column_stack([X_price.ravel(), roast_nums])
                preds = model2.predict(X2)
            else:
                raise RuntimeError("No available model to make predictions")
        else:
            preds[remaining] = model1.predict(X_price[remaining])

    return preds
