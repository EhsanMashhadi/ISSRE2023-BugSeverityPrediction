package software.ehsan.severityprediction.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GitUtil {

    public static String checkoutDJ4(String projectName, String projectVersion, boolean buggy) throws IOException {
        String command = "";
        if (buggy) {
            command = "git checkout " + "D4J_" + projectName + "_" + projectVersion + "_BUGGY_VERSION";
        } else {
            command = "git checkout " + "D4J_" + projectName + "_" + projectVersion + "_FIXED_VERSION";
        }
        String projectRepoBasePath = Constants.D4J_PROJECTS_ROOT;
        return runCommand(command, projectRepoBasePath, projectName + projectVersion + "b");
    }

    public static String checkoutBugsJar(String projectName, String branchName, boolean buggy) throws IOException {
        String command = "";
        if (buggy) {
            command = "git checkout " + branchName;
        } else {
            command = "git checkout " + branchName.split("_")[branchName.split("_").length - 1];
        }
        String projectRepoBasePath = Constants.BUGS_JAR_PROJECTS_ROOT;
        return runCommand(command, projectRepoBasePath, projectName);
    }

    private static String runCommand(String command, String projectRepoBasePath, String projectName) throws IOException {
        Process process = Runtime.getRuntime().exec(command, null, new File(projectRepoBasePath, projectName));
        StringBuilder result = new StringBuilder();
        BufferedReader in = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String inputLine;
        while ((inputLine = in.readLine()) != null) {
            result.append(inputLine);
        }
        in.close();
        return result.toString();
    }

    public static List<String> getBugsJarBuggyFiles(String projectName, String branchName) throws IOException {

        String command = "git diff " + branchName.split("_")[branchName.split("_").length - 1] + " --name-only";
        String projectRepoBasePath = Constants.BUGS_JAR_PROJECTS_ROOT;
        return getChangedFiles(command, projectRepoBasePath, projectName);
    }

    public static List<String> getD4JBuggyFiles(String projectName, String projectVersion) throws IOException {
        String command = "git diff " + "D4J_" + projectName + "_" + projectVersion + "_FIXED_VERSION" + " --name-only";
        String projectRepoBasePath = Constants.D4J_PROJECTS_ROOT;
        return getChangedFiles(command, projectRepoBasePath, projectName + projectVersion + "b");
    }

    private static List<String> getChangedFiles(String command, String projectRepoBasePath, String projectName) throws IOException {
        Process process = Runtime.getRuntime().exec(command, null, new File(projectRepoBasePath, projectName));
        List<String> result = new ArrayList<>();
        BufferedReader in = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String inputLine;
        while ((inputLine = in.readLine()) != null) {
            if (inputLine.endsWith(".java")) {
                result.add(inputLine);
            }
        }
        in.close();
        return result;
    }
}
