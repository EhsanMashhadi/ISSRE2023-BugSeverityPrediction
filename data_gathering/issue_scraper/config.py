import os as os
from pathlib import Path

d4j_projects = ["Chart", "Cli", "Closure", "Codec", "Collections", "Compress", "Csv", "Gson", "JacksonCore",
                "JacksonDatabind", "JacksonXml", "Jsoup", "JxPath", "Lang", "Math", "Mockito", "Time"]

d4j_sub_projects = ["JxPath"]

bugs_jar_projects = ["accumulo", 'camel', 'commons-math', 'flink', 'jackrabbit-oak', 'logging-log4j2', 'maven',
                     'wicket']

bugs_jar_sub_projects = ["jackrabbit-oak"]

d4j_severity_groups = {
    "low": ["Low", "Trivial", "Minor", "7", "8", "9", "['Type-Defect', 'Priority-Low']"],
    "medium": ["Medium", "5", "['Type-Defect', 'Priority-Medium']",
               "['Type-Defect', 'Priority-Medium', 'Restrict-AddIssueComment-Nobody']",
               "['Type-Enhancement', 'Priority-Medium']", "['Type-Defect', 'Priority-Medium', 'ES5']",
               "['Type-Defect', 'Priority-Medium', 'Component-Parser']"
               ],
    "high": ["Major", "High", "3", "['Type-Defect', 'Priority-High']"],
    "critical": ["Critical", "Blocker", "['Type-Defect', 'Priority-Critical']"],
    "not_valid": ["['Type-Enhancement']"]
}

GOOGLE_URL = "googleapis"
JIRA_URL = "jira"
SOURCE_FORGE_URL = "sourceforge"

PROJECT_NAME_CSV_COL = "ProjectName"
PROJECT_VERSION_CSV_COL = "ProjectVersion"
PROJECT_IssueTracker_CSV_COL = "IssueTracker"
PROJECT_URL_CSV_COL = "URL"
PROJECT_Severity_CSV_COL = "Severity"


def get_project_root() -> Path:
    return Path(__file__).parent.parent


ROOT_DIR = get_project_root()
DATA_DIR = os.path.join(ROOT_DIR, "data")
D4J_FILE = "d4j_bugs.csv"
D4J_FOLDER = "d4j/"
BUGS_JAR_FILE = "bugs_jar_bugs.csv"
BUGS_JAR_FOLDER = os.path.join(ROOT_DIR, "bugs-dot-jar/")
