import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from data.eu_loader import load_eu_data

class EUClient(fl.client.NumPyClient):
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = load_eu_data()

        print("EU CLIENT INIT labels:",
              np.unique(self.y_train, return_counts=True))

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs"
        )

        # 🚨 DO NOT FIT HERE

    def get_parameters(self, config):
        if not hasattr(self.model, "coef_"):
            n_features = self.X_train.shape[1]
            self.model.coef_ = np.zeros((1, n_features))
            self.model.intercept_ = np.zeros(1)
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.fit(self.X_train, self.y_train)

        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = self.model.score(self.X_test, self.y_test)
        return float(acc), len(self.X_test), {"accuracy": acc}


if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=EUClient(),
    )
