from sklearn.ensemble import RandomForestClassifier


class RandomForest:

    @staticmethod
    def train_test(x_train, y_train, x_test, random_state):
        random_forest = RandomForestClassifier(random_state=random_state, n_jobs=-1, class_weight='balanced')
        random_forest.fit(X=x_train, y=y_train)
        prob = random_forest.predict_proba(X=x_test)
        return random_forest.predict(X=x_test), prob
