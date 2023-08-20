from sklearn.neural_network import MLPClassifier


class Mlp:

    @staticmethod
    def train_test(x_train, y_train, x_test, random_state):
        mlp = MLPClassifier(random_state=random_state)
        mlp.fit(X=x_train, y=y_train)
        prob = mlp.predict_proba(X=x_test)
        return mlp.predict(X=x_test), prob
