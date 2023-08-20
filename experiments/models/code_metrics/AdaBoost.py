from sklearn.ensemble import AdaBoostClassifier


class AdaBoost:

    @staticmethod
    def train_test(x_train, y_train, x_test, random_state):
        ada_boost = AdaBoostClassifier(random_state=random_state)
        ada_boost.fit(X=x_train, y=y_train)
        prob = ada_boost.predict_proba(X=x_test)
        return ada_boost.predict(X=x_test), prob
