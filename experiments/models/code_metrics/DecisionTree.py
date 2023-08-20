from sklearn.tree import DecisionTreeClassifier


class DecisionTree:

    @staticmethod
    def train_test(x_train, y_train, x_test, random_state):
        decision_tree = DecisionTreeClassifier(random_state=random_state)
        decision_tree.fit(X=x_train, y=y_train)
        prob = decision_tree.predict_proba(X=x_test)
        return decision_tree.predict(X=x_test), prob
