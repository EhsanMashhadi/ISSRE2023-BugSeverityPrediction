from sklearn.neighbors import KNeighborsClassifier


class KNN:

    @staticmethod
    def train_test(x_train, y_train, x_test):
        knn = KNeighborsClassifier()
        knn.fit(X=x_train, y=y_train)
        prob = knn.predict_proba(X=x_test)
        return knn.predict(X=x_test), prob
