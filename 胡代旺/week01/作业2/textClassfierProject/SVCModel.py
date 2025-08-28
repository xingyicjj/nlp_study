from ModelStrategy import ModelStrategy
from sklearn.svm import SVC


class SVCModel(ModelStrategy):
    def __init__(self, **kwargs):
        self.model = SVC(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
