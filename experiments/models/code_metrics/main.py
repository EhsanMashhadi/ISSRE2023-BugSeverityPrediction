import logging
import os

import pandas as pd

from models.code_metrics.AdaBoost import AdaBoost
from models.code_metrics.DecisionTree import DecisionTree
from models.code_metrics.KNN import KNN
from models.code_metrics.Mlp import Mlp
from models.code_metrics.NaiveBayes import NaiveBayes
from models.code_metrics.RandomForest import RandomForest
from models.code_metrics.SVM import SVM
from models.code_metrics.XgBoost import XgBoost
from models.evaluation.evaluation import evaluatelog_result
from util.Constants import DATASET_DIR

logger = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.makedirs("log")
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
file_handler = logging.FileHandler(os.path.join("log", "log-{}.txt".format("code_metrics")))
logger.addHandler(file_handler)

random_state = 42


def decision_tree():
    result, prob = DecisionTree.train_test(x_train=x_train, y_train=y_train, x_test=x_test, random_state=random_state)
    evaluatelog_result(y_test, result, prob, "Decision Tree", logger)


def random_forest():
    result, prob = RandomForest.train_test(x_train=x_train, y_train=y_train, x_test=x_test, random_state=random_state)
    evaluatelog_result(y_test, result, prob, "Random Forest", logger)


def svc():
    result, prob = SVM.train_test(x_train=x_train, y_train=y_train, x_test=x_test, random_state=random_state)
    evaluatelog_result(y_test, result, prob, "SVM", logger)


def naive_bayes():
    result, prob = NaiveBayes.train_test(x_train=x_train, y_train=y_train, x_test=x_test)
    evaluatelog_result(y_test, result, prob, "Naive Bayes", logger)


def knn():
    result, prob = KNN.train_test(x_train=x_train, y_train=y_train, x_test=x_test)
    evaluatelog_result(y_test, result, prob, "KNN", logger)


def xgboost():
    result, prob = XgBoost.train_test(x_train=x_train, y_train=y_train, x_test=x_test, random_state=random_state)
    evaluatelog_result(y_test, result, prob, "XGBoost", logger)


def adaboost():
    result, prob = AdaBoost.train_test(x_train=x_train, y_train=y_train, x_test=x_test, random_state=random_state)
    evaluatelog_result(y_test, result, prob, "Ada Boost", logger)


def mlp():
    result, prob = Mlp.train_test(x_train=x_train, y_train=y_train, x_test=x_test, random_state=random_state)
    evaluatelog_result(y_test, result, prob, "MLP", logger)


if __name__ == '__main__':
    cols = ['lc', 'pi', 'ma', 'nbd', 'ml', 'd', 'mi', 'fo', 'r', 'e']

    train = pd.read_csv(os.path.join(DATASET_DIR, "train_scaled.csv"))
    test = pd.read_csv(os.path.join(DATASET_DIR, "test_scaled.csv"))

    y_train = train["label"]
    y_test = test["label"]

    x_train = train[cols]
    x_test = test[cols]

    decision_tree()
    random_forest()
    svc()
    naive_bayes()
    knn()
    xgboost()
    adaboost()
    mlp()
