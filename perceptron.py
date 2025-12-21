from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class PerceptronModel :
    def __init__ (self):
        self.model = Perceptron()

    def fit (self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self,X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, y_true, y_pred):

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

        return metrics


