from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RandomForest:
    """Clase para definir, entrenar y evaluar un modelo RandomForest."""

    def __init__(self, n_estimators=100, max_features='auto', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True,
                 criterion='gini', random_state=42):
        """Inicializa el modelo RandomForest con los parámetros dados."""
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            criterion=self.criterion,
            random_state=self.random_state
        )

    def fit(self, X_train, y_train, X_valid, y_valid):
        """Entrena el modelo RandomForest con los datos de entrenamiento y valida con los datos de validación."""
        X_train_shape = X_train.shape
        X_valid_shape = X_valid.shape
        if len(X_train_shape) == 3:
            X_train = X_train.reshape(-1, X_train_shape[-1])
            X_valid = X_valid.reshape(-1, X_valid_shape[-1])
        self.model.fit(X_train, y_train)
        val_predictions = self.model.predict(X_valid)
        acc = accuracy_score(y_valid, val_predictions)
        print(f"Accuracy de validación: {acc}")
        return acc
    
    def predict(self, X):
        """Predice las etiquetas para los datos dados."""
        X_shape = X.shape
        if len(X_shape) == 3:
            X = X.reshape(-1, X_shape[-1])
        return self.model.predict(X)