import os
import subprocess

import pandas as pd

import config


def get_project_urls(project_name):
    result = subprocess.run(['defects4j', 'query', '-p', project_name, "-q", "report.url"],
                            stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8").strip().split("\n")


def get_project_ids(project_name):
    result = subprocess.run(['defects4j', 'query', '-p', project_name],
                            stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8").strip().split("\n")


def checkout_project(project_name, version, output):
    result = subprocess.run(['defects4j', 'checkout', "-p", project_name, "-v", version, "-w", output],
                            stdout=subprocess.PIPE)
    return result.returncode


def get_buggy_files(directory):
    result = subprocess.run(['defects4j', 'export', '-p', 'classes.modified'], cwd=directory,
                            stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8").strip().split("\n")


def compile_project(directory):
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
    result = subprocess.run(['defects4j', 'compile'], cwd=directory,
                            stdout=subprocess.PIPE)
    return result.returncode


def get_project_target(directory):
    result = subprocess.run(['defects4j', 'export', '-p', 'dir.bin.classes'], cwd=directory,
                            stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8").strip().split("\n")[0]


def get_project_classpath(directory):
    result = subprocess.run(['defects4j', 'export', '-p', 'cp.compile'], cwd=directory,
                            stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8").strip()


def get_src_classes(base_directory):
    result = subprocess.run(['defects4j', 'export', "-p", "dir.src.classes"], cwd=base_directory,
                            stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8").strip().split("\n")


def get_buggy_commit(project_name, project_version):
    result = subprocess.run(['defects4j', 'query', '-p', project_name, "-q", "revision.id.buggy"],
                            stdout=subprocess.PIPE)
    row = result.stdout.decode("utf-8").strip().split("\n")
    for id_commit in row:
        version = id_commit.split(",")[0]
        buggy_commit = id_commit.split(",")[1]

        if int(version) == project_version:
            return buggy_commit


def unify_d4j():
    df = pd.read_csv(os.path.join(config.DATA_DIR, config.D4J_FILE))
    df['Severity'].replace(
        config.d4j_severity_groups["low"], "Low", inplace=True)
    df['Severity'].replace(
        config.d4j_severity_groups["medium"], "Medium", inplace=True)
    df['Severity'].replace(
        config.d4j_severity_groups["high"], "High", inplace=True)
    df['Severity'].replace(
        config.d4j_severity_groups["critical"], "Critical", inplace=True)
    df['Severity'].replace(
        config.d4j_severity_groups["not_valid"], "", inplace=True)
    return df


def checkout_projects():
    df = pd.read_csv(os.path.join(config.DATA_DIR, config.D4J_FILE))
    for index, row in df.iterrows():
        project = row[config.PROJECT_NAME_CSV_COL]
        version = row[config.PROJECT_VERSION_CSV_COL]
        directory = os.path.join(config.ROOT_DIR, config.D4J_FOLDER, "{}{}b".format(project, int(version)))
        severity = row[config.PROJECT_Severity_CSV_COL]
        if not pd.isna(severity):
            result = checkout_project(project_name=project, version="{}b".format(version), output=directory)
            assert result == 0
