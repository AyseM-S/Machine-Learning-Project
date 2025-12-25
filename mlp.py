from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class MLP_Model:

    def __init__(self, hidden_layers =(64,32),random_state=42):
        self.model = MLPClassifier(
            hidden_layer_sizes= hidden_layers,
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=random_state
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "Confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

        return metrics




