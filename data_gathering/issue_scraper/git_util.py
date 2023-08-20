import os.path
import subprocess

from config import BUGS_JAR_FOLDER


def checkout_master_branch(path: str):
    result = subprocess.run(["git", "checkout", "master"], cwd=path, stdout=subprocess.PIPE)
    output = result.stdout.decode("utf-8").strip().split("\n")
    return output


def get_bugs_jar_all_branches(project_name: str):
    path = os.path.join(BUGS_JAR_FOLDER,project_name)
    result = subprocess.run(["git", "branch", "-a"], cwd=path, stdout=subprocess.PIPE)
    branches_name = result.stdout.decode("utf-8").strip().split("\n")
    filtered_branches_name = []
    for branch_name in branches_name:
        if "remotes/origin/bugs-dot-jar" in branch_name:
            filtered_branches_name.append(branch_name.strip())
    return filtered_branches_name


def checkout_bugs_jar(project_name, branch_name, buggy=True):
    path = "/home/ehsan/Workspace/java/bugsjar/bugs-dot-jar/" + project_name
    result = None
    if buggy:
        subprocess.run(["git", "checkout", "--", "."], cwd=path, stdout=subprocess.PIPE)
        result = subprocess.run(["git", "clean", "-f", "-d"], cwd=path, stdout=subprocess.PIPE)
        result = subprocess.run(["git", "checkout", branch_name], cwd=path, stdout=subprocess.PIPE)

    return result


def build_project_accumulo(project_name):
    path = "/home/ehsan/Workspace/java/bugsjar/bugs-dot-jar/" + project_name
    # result = subprocess.run(["mvn", "clean", "install", "-Drat.numUnapprovedLicenses=100", "-Dmaven.test.skip"],
    #                         cwd=path,
    #                         stdout=subprocess.PIPE)
    result = subprocess.run(["mvn", "-T1C", "clean", "compiler:compile"],
                            cwd=path,
                            stdout=subprocess.PIPE)
    return result


def build_project_camel(project_name):
    path = "/home/ehsan/Workspace/java/bugsjar/bugs-dot-jar/" + project_name
    result = subprocess.run(["mvn", "clean", "install", "-Pfastinstall"],
                            cwd=path,
                            stdout=subprocess.PIPE)
    return result


def build_project_math(project_name):
    path = "/home/ehsan/Workspace/java/bugsjar/bugs-dot-jar/" + project_name
    result = subprocess.run(["ant", "compile", "-Dmaven.test.skip=true"],
                            cwd=path,
                            stdout=subprocess.PIPE)
    return result


def build_project_flink(project_name):
    path = "/home/ehsan/Workspace/java/bugsjar/bugs-dot-jar/" + project_name
    result = subprocess.run(["mvn", "-T1C", "clean", "compiler:compile"],
                            cwd=path,
                            stdout=subprocess.PIPE)
    return result


def build_project_jackrabbit(project_name):
    path = "/home/ehsan/Workspace/java/bugsjar/bugs-dot-jar/" + project_name
    result = subprocess.run(["mvn", "-T1C", "clean", "compiler:compile"],
                            cwd=path,
                            stdout=subprocess.PIPE)
    return result


def build_project_logging(project_name):
    path = "/home/ehsan/Workspace/java/bugsjar/bugs-dot-jar/" + project_name
    result = subprocess.run(
        ["mvn", "-T1C", "clean", "install", "-Drat.numUnapprovedLicenses=100", "-Dmaven.test.skip"],
        cwd=path,
        stdout=subprocess.PIPE)
    return result


def build_project_maven(project_name):
    path = "/home/ehsan/Workspace/java/bugsjar/bugs-dot-jar/" + project_name
    result = subprocess.run(
        ["mvn", "-T1C", "-Drat.numUnapprovedLicenses=100", "clean", "install", "-Dmaven.test.skip"],
        cwd=path,
        stdout=subprocess.PIPE)
    return result


def build(project_name):
    path = "/home/ehsan/Workspace/java/bugsjar/bugs-dot-jar/" + project_name
    result = subprocess.run(
        ["mvn", "clean", "compile"],
        cwd=path,
        stdout=subprocess.PIPE)
    return result
