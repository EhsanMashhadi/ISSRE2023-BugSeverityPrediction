package software.ehsan.severityprediction.method_extractor;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.body.MethodDeclaration;
import software.ehsan.severityprediction.model.Bug;
import software.ehsan.severityprediction.model.D4JBug;
import software.ehsan.severityprediction.model.Method;
import software.ehsan.severityprediction.utils.Constants;
import software.ehsan.severityprediction.utils.GitUtil;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

public class D4JMethodExtractor implements MethodExtractor {
    @Override
    public List<List<Method>> extractMethods(List<Bug> bugs) {

        List<Method> buggyMethods = new ArrayList<>();
        List<Method> nonBuggyMethods = new ArrayList<>();
        List<Method> fixedMethods = new ArrayList<>();

        logger.log(Level.INFO, "Defects4J method extractor started. This make take several minutes...");
        logger.log(Level.INFO, "----------------------------------------------------------------------");

        try {
            for (Bug bug : bugs) {

                D4JBug d4JBug = (D4JBug) bug;
                String projectName = d4JBug.getProjectName();
                String projectVersion = d4JBug.getProjectVersion();
                String severity = d4JBug.getSeverity();

                logger.log(Level.INFO, "Project name: %s, Project version: %s, Severity: %s".formatted(projectName, projectVersion, severity));

                if (!severity.isEmpty()) {
                    GitUtil.checkoutDJ4(projectName, projectVersion, true);
                    List<String> changedFiles = GitUtil.getD4JBuggyFiles(projectName, projectVersion);
                    String projectRepoBasePath = Constants.D4J_PROJECTS_ROOT;
                    for (String changedFile : changedFiles) {
                        GitUtil.checkoutDJ4(projectName, projectVersion, true);
                        Path path = Paths.get(projectRepoBasePath, projectName + projectVersion + "b", changedFile);
                        List<MethodDeclaration> buggyFileMethods = JavaParser.parse(path).findAll(MethodDeclaration.class);
                        GitUtil.checkoutDJ4(projectName, projectVersion, false);
                        List<MethodDeclaration> fixedFileMethods = JavaParser.parse(path).findAll(MethodDeclaration.class);
                        for (MethodDeclaration suspiciousMethod : buggyFileMethods) {
                            if (!fixedFileMethods.contains(suspiciousMethod)) {
                                buggyMethods.add(new Method(true, bug, changedFile, suspiciousMethod.getBegin().get().line, suspiciousMethod.getEnd().get().line, suspiciousMethod.toString()));
                            } else {
                                nonBuggyMethods.add(new Method(false, bug, changedFile, suspiciousMethod.getBegin().get().line, suspiciousMethod.getEnd().get().line, suspiciousMethod.toString()));
                            }
                        }
                        for (MethodDeclaration fixedMethod : fixedFileMethods) {
                            if (!buggyFileMethods.contains(fixedMethod)) {
                                fixedMethods.add(new Method(false, bug, changedFile, fixedMethod.getBegin().get().line, fixedMethod.getEnd().get().line, fixedMethod.toString()));
                            }
                        }
                    }
                }
            }

            ArrayList<List<Method>> methods = new ArrayList<>();
            methods.add(buggyMethods);
            methods.add(nonBuggyMethods);
            methods.add(fixedMethods);
            return methods;
        } catch (IOException e) {
            logger.log(Level.SEVERE, e.toString());
            return null;
        }
    }
}