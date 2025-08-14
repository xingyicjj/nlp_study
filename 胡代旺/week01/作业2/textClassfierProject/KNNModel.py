from ModelStrategy import ModelStrategy
from sklearn.neighbors import KNeighborsClassifier


class KNNModel(ModelStrategy):
    def __init__(self, **kwargs):
        self.model = KNeighborsClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)