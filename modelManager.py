from sklearn.model_selection import train_test_split
import Perceptron.py
import MLP.py
import DecisionTree.py


class ModelManager:

    def __init__(self, test_ratio=0.2, random_state=42):
        self.test_ratio = test_ratio
        self.random_state = random_state

        # Adding models to dictionary
        self.available_models = {
            "Perceptron": Perceptron,
            "MLP": MLP,
            "DecisionTree": DecisionTree
        }
    
    # Changing test ratio
    def set_test_ratio(self, ratio):
        self.test_ratio = ratio


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

        # Modeli oluştur
        ModelClass = self.available_models[selected_model_name]
        model = ModelClass()

        # Eğit
        model.fit(X_train, y_train)

        # Tahmin
        y_pred = model.predict(X_test)

        # Değerlendir
        results = model.evaluate(y_test, y_pred)

        return results

    def run_multiple(self, X, y, selected_models:list):
        """
        Kullanıcı birden fazla modeli aynı anda seçerse
        örnek: ["Perceptron", "DecisionTree"]
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

