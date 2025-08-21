from KNNModel import KNNModel
from RandomForestClassifier import RandomForest
from SVCModel import SVCModel


class ModelFactory:
    _models = {
        'svc': SVCModel,
        'knn': KNNModel,
        'forest': RandomForest
    }

    @classmethod
    def create_model(cls, model_type, **kwargs):
        model_class = cls._models.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model_class(**kwargs)
