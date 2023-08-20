import json
import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from util.Constants import DATA_DIR


class Bug:
    def __init__(self, project_name, project_version, severity, code, code_comment, code_no_comment, lc, pi, ma, nbd,
                 ml, d, mi, fo, r, e):
        self.project_name = project_name
        self.project_version = project_version
        self.label = severity
        self.code = code
        self.code_comment = code_comment
        self.code_no_comment = code_no_comment
        self.lc = lc
        self.pi = pi
        self.ma = ma
        self.nbd = nbd
        self.ml = ml
        self.d = d
        self.mi = mi
        self.fo = fo
        self.r = r
        self.e = e


def split_dataset():
    d4j = pd.read_csv(os.path.join(DATA_DIR, "d4j_methods_sc_metrics_comments.csv"))
    new_d4j = categorical_to_number_d4j(d4j)
    bugs_jar = pd.read_csv(os.path.join(DATA_DIR, "bugsjar_methods_sc_metrics_comments.csv"))
    new_bugs_jar = categorical_to_number_bugsjar(bugs_jar)
    bugs = []
    bugs.extend(create_bugs(new_d4j))
    bugs.extend(create_bugs(new_bugs_jar))

    df_bugs = pd.DataFrame(bugs)
    df_bugs.drop_duplicates(keep='first', inplace=True)

    train, test = train_test_split(df_bugs, test_size=0.15, random_state=666, shuffle=True)
    train, val = train_test_split(train, test_size=0.15, random_state=666, shuffle=True)

    cols = ['lc', 'pi', 'ma', 'nbd', 'ml', 'd', 'mi', 'fo', 'r', 'e']
    scaler = RobustScaler()
    train[cols] = scaler.fit_transform(train[cols])
    test[cols] = scaler.transform(test[cols])
    val[cols] = scaler.transform(val[cols])

    write_bugs(train, "train_scaled")
    write_bugs(val, "valid_scaled")
    write_bugs(test, "test_scaled")


def create_bugs(df):
    bugs = []
    for index, row in df.iterrows():
        if row["IsBuggy"]:
            bug = Bug(project_name=row["ProjectName"], project_version=row["ProjectVersion"], severity=row["Severity"],
                      code=row["SourceCode"], code_comment=row["CodeComment"],
                      code_no_comment=row["CodeNoComment"], lc=row["LC"], pi=row["PI"], ma=row["MA"],
                      nbd=row["NBD"], ml=row["ML"], d=row["D"], mi=row["MI"], fo=row["FO"], r=row["R"], e=row["E"])
            bugs.append(bug.__dict__)
    return bugs


def write_bugs(bugs, name):
    with open("{}.jsonl".format(name), 'w') as f:
        for bug in bugs.to_dict("records"):
            f.write(json.dumps(bug) + "\n")
    df = pd.DataFrame(bugs)
    df.to_csv("{}.csv".format(name), index=False)


def categorical_to_number_d4j(df):
    df['Severity'].replace("Critical", 0, inplace=True)
    df['Severity'].replace("High", 1, inplace=True)
    df['Severity'].replace("Medium", 2, inplace=True)
    df['Severity'].replace("Low", 3, inplace=True)

    return df


def categorical_to_number_bugsjar(df):
    df['Severity'].replace("Blocker", 0, inplace=True)
    df['Severity'].replace("Critical", 0, inplace=True)
    df['Severity'].replace("Major", 1, inplace=True)
    df['Severity'].replace("Minor", 3, inplace=True)
    df['Severity'].replace("Trivial", 3, inplace=True)
    return df


if __name__ == '__main__':
    split_dataset()
