package software.ehsan.severityprediction.utils;

import java.io.File;

public class Constants {

    public static final String ROOT_DIR = System.getProperty("user.dir");
    public static final String DATA_DIR = new File(ROOT_DIR).getParent() + "/data";

    public static final String D4J_DATA_FILE = new File(DATA_DIR,"d4j_bugs.csv").getAbsolutePath();

    public static final String D4J_PROJECTS_ROOT = new File(ROOT_DIR).getParent()+"/d4j";

    public static final String BUGS_JAR_PROJECTS_ROOT = new File(ROOT_DIR).getParent()+"/bugs-dot-jar";
    public static final String BUGS_JAR_DATA_FILE = new File(DATA_DIR,"bugs_jar_bugs.csv").getAbsolutePath();

    //--------------------------------------------------------------------------------

    public static final String METHOD_IS_BUGGY_CSV_COL = "IsBuggy";
    public static final String PROJECT_NAME_CSV_COL = "ProjectName";
    public static final String PROJECT_VERSION_CSV_COL = "ProjectVersion";
    public static final String PROJECT_ISSUE_TRACKER_CSV_COL = "IssueTracker";
    public static final String PROJECT_URL_CSV_COL = "Url";
    public static final String SEVERITY_CSV_COL = "Severity";
    public static final String CLASS_NAME_CSV_COL = "ClassName";
    public static final String START_LINE_CSV_COL = "StartLine";
    public static final String END_LINE_CSV_COL = "EndLine";
    public static final String SOURCE_CODE_CSV_COL = "SourceCode";
    public static final String DEFECTS4J_BUGGY_METHOD_FILE = "d4j_methods_buggy.csv";
    public static final String DEFECTS4J_NON_BUGGY_METHOD_FILE = "d4j_methods_nonbuggy.csv";
    public static final String DEFECTS4J_FIXED_METHOD_FILE = "d4j_methods_fixed.csv";
    public static final String BUGS_JAR_BUGGY_METHOD_FILE = "bugsjar_methods_buggy.csv";
    public static final String BUGS_JAR_NON_BUGGY_METHOD_FILE = "bugsjar_methods_nonbuggy.csv";
    public static final String BUGS_JAR_FIXED_METHOD_FILE = "bugsjar_methods_fixed.csv";
    public static final String DEFECTS4J_METRICS_FILE = "d4j_methods_sc_metrics.csv";
    public static final String BUGS_JAR_METRICS_FILE = "bugsjar_methods_sc_metrics.csv";

    //--------------------------------------------------------------------------------
    public static final String SLOC_COL = "LC";
    public static final String PROXY_INDENTATION_COL = "PI";
    public static final String MCCABE_COL = "MA";
    public static final String MCCLURE_COL = "ML";
    public static final String NESTED_BLOCK_DEPTH_COL = "NBD";
    public static final String DIFFICULTY_COL = "D";
    public static final String MAINTAINABILITY_COL = "MI";
    public static final String FAN_OUT_COL = "FO";
    public static final String READABILITY_COL = "R";
    public static final String EFFORT_COL = "E";

    //---------------------------------------------------------------------------------
}
