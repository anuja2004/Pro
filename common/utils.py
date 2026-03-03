import numpy as np
from sklearn.metrics import log_loss, accuracy_score

def get_class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))

def evaluate_model(model, X, y):
    probs = model.predict_proba(X)
    preds = model.predict(X)

    loss = log_loss(y, probs)
    acc = accuracy_score(y, preds)

    return loss, acc
