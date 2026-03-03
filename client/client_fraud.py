import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression

from data.fraud_loader import load_fraud_data
from common.evaluate import evaluate_model


class FraudClient(fl.client.NumPyClient):
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = load_fraud_data()

        unique, counts = np.unique(self.y_train, return_counts=True)
        print("FraudClient TRAIN labels:", dict(zip(unique, counts)))

        if len(unique) < 2:
            raise ValueError(
                "Training data contains only one class. "
                "Federated learning requires both classes."
            )

        self.model = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="lbfgs"
        )

        # ---- SAFE dummy initialization ----
        idx_0 = np.where(self.y_train == 0)[0][0]
        idx_1 = np.where(self.y_train == 1)[0][0]

        X_init = np.vstack([
            self.X_train[idx_0],
            self.X_train[idx_1],
        ])
        y_init = np.array([0, 1])

        self.model.fit(X_init, y_init)


    def get_parameters(self, config=None):
        return [
            self.model.coef_.astype(np.float32),
            self.model.intercept_.astype(np.float32),
        ]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.fit(self.X_train, self.y_train)

        return self.get_parameters(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss, accuracy = evaluate_model(
            self.model,
            self.X_test,
            self.y_test
        )

        return loss, len(self.X_test), {"accuracy": accuracy}


if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FraudClient(),
    )
