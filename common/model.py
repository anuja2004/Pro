from sklearn.linear_model import LogisticRegression

def get_model():
    return LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="lbfgs",
    )
