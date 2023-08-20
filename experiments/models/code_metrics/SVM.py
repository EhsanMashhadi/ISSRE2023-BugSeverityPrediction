from sklearn.svm import SVC


class SVM:

    @staticmethod
    def train_test(x_train, y_train, x_test, random_state):
        svc = SVC(kernel="rbf", probability=True, random_state=random_state)
        svc.fit(X=x_train, y=y_train)
        prob = svc.predict_proba(X=x_test)
        return svc.predict(X=x_test), prob
