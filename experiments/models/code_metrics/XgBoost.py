from xgboost import XGBClassifier


class XgBoost:
    @staticmethod
    def train_test(x_train, y_train, x_test, random_state):
        xg_boost = XGBClassifier(random_state=random_state)
        xg_boost.fit(X=x_train, y=y_train)
        prob = xg_boost.predict_proba(X=x_test)
        return xg_boost.predict(X=x_test), prob
