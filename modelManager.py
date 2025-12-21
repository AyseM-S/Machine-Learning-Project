from sklearn.model_selection import train_test_split
from perceptron import PerceptronModel
from mlp import MLP_Model
from decisionTree import DecisionTreeModel


class ModelManager:

    def __init__(self, test_ratio=0.2, random_state=42):
        self.test_ratio = test_ratio
        self.random_state = random_state

        # Adding models to dictionary
        self.available_models = {
            "Perceptron": PerceptronModel,
            "MLP": MLP_Model,
            "DecisionTree": DecisionTreeModel
        }
    
    # Changing test ratio
    def set_test_ratio(self, ratio):
        if ratio <= 0 or ratio >= 1:
            print("Test ratio must be between 0 and 1.")
        return None

    self.test_ratio = 1 - split_ratio



    def split(self, X, y):
        test = self.test_ratio
        seed = self.random_state

        X_train, X_test, y_train, y_test = train_test_split(
           X, y,
           test_size=test,
           random_state=seed,
           stratify=y
    )

        return X_train, X_test, y_train, y_test
    
    # Selecting the model
    def run(self, X, y, selected_model_name):
        """
        selected_model_name: "Perceptron" | "MLP" | "DecisionTree"
        """

        if selected_model_name not in self.available_models:
            print("The model not found")
            return None 
        
        # Splitting data 
        X_train, X_test, y_train, y_test = self.split(X, y)

        # Creating the model
        ModelClass = self.available_models[selected_model_name]

        if selected_model_name == "MLP":
        if hidden_layers is None:
            # There is not value from GUI -> default
            model = ModelClass()
        else:
            # "128,64,32" → (128, 64, 32)
            layer_tuple = tuple(int(x.strip()) for x in hidden_layers.split(","))
            model = ModelClass(hidden_layers=layer_tuple)
    else:
        model = ModelClass()


        # Training
        model.fit(X_train, y_train)

        # Predicition
        y_pred = model.predict(X_test)

        # Evaluating
        results = model.evaluate(y_test, y_pred)

        return results

    def run_multiple(self, X, y, selected_models:list):
        """
        If the user select more than one model
        Example: ["Perceptron", "DecisionTree"]
        """

        X_train, X_test, y_train, y_test = self.split(X, y)

        results = {}

        for name in selected_models:
            if name in self.available_models:
                ModelClass = self.available_models[name]
                model = ModelClass()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = model.evaluate(y_test, y_pred)

        return results



