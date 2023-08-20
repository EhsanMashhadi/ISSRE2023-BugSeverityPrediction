from sklearn.naive_bayes import GaussianNB


class NaiveBayes:

    @staticmethod
    def train_test(x_train, y_train, x_test):
        naive_bayes = GaussianNB()
        naive_bayes.fit(X=x_train, y=y_train)
        prob = naive_bayes.predict_proba(X=x_test)
        return naive_bayes.predict(X=x_test), prob
