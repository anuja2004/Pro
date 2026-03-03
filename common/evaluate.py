import numpy as np
from sklearn.metrics import log_loss, accuracy_score


def evaluate_model(model, X, y):
    """
    Evaluate a trained sklearn model.

    Returns
    -------
    loss : float
        Log loss (cross-entropy)
    accuracy : float
        Classification accuracy
    """

    # Probability predictions (required for log loss)
    probs = model.predict_proba(X)

    # Class predictions
    preds = model.predict(X)

    # Compute metrics
    loss = log_loss(y, probs)
    acc = accuracy_score(y, preds)

    return loss, acc
