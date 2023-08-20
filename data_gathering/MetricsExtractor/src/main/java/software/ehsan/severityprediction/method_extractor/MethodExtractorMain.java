package software.ehsan.severityprediction.method_extractor;

import de.siegmar.fastcsv.reader.CloseableIterator;
import de.siegmar.fastcsv.reader.CsvReader;
import de.siegmar.fastcsv.reader.CsvRow;
import de.siegmar.fastcsv.writer.CsvWriter;
import software.ehsan.severityprediction.model.Bug;
import software.ehsan.severityprediction.model.BugsJarBug;
import software.ehsan.severityprediction.model.D4JBug;
import software.ehsan.severityprediction.model.Method;
import software.ehsan.severityprediction.utils.Constants;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

public class MethodExtractorMain {


    public static void main(String[] args) throws IOException {
        MethodExtractorMain methodExtractorMain = new MethodExtractorMain();
//        methodExtractorMain.extractDefect4JMethods();
        methodExtractorMain.extractBugJarMethods();
    }

    public void extractDefect4JMethods() throws IOException {

        CsvReader csvReader = CsvReader.builder().build(Paths.get(Constants.D4J_DATA_FILE));
        CloseableIterator<CsvRow> csvRow = csvReader.iterator();
        csvRow.next();
        List<Bug> bugs = csvReader.stream().map(row -> new D4JBug(row.getField(0), row.getField(1), row.getField(2), row.getField(3), row.getField(4))).collect(Collectors.toList());
        csvRow.close();

        CsvWriter buggyMethodsCsvWriter = CsvWriter.builder().build(Paths.get(Constants.DATA_DIR, Constants.DEFECTS4J_BUGGY_METHOD_FILE));
        CsvWriter nonBuggyMethodsCsvWriter = CsvWriter.builder().build(Paths.get(Constants.DATA_DIR, Constants.DEFECTS4J_NON_BUGGY_METHOD_FILE));
        CsvWriter fixedMethodsCsvWriter = CsvWriter.builder().build(Paths.get(Constants.DATA_DIR, Constants.DEFECTS4J_FIXED_METHOD_FILE));

        buggyMethodsCsvWriter.writeRow(Constants.PROJECT_NAME_CSV_COL, Constants.PROJECT_VERSION_CSV_COL, Constants.PROJECT_ISSUE_TRACKER_CSV_COL, Constants.PROJECT_URL_CSV_COL, Constants.SEVERITY_CSV_COL, Constants.CLASS_NAME_CSV_COL, Constants.START_LINE_CSV_COL, Constants.END_LINE_CSV_COL, Constants.SOURCE_CODE_CSV_COL);
        nonBuggyMethodsCsvWriter.writeRow(Constants.PROJECT_NAME_CSV_COL, Constants.PROJECT_VERSION_CSV_COL, Constants.PROJECT_ISSUE_TRACKER_CSV_COL, Constants.PROJECT_URL_CSV_COL, Constants.SEVERITY_CSV_COL, Constants.CLASS_NAME_CSV_COL, Constants.START_LINE_CSV_COL, Constants.END_LINE_CSV_COL, Constants.SOURCE_CODE_CSV_COL);
        fixedMethodsCsvWriter.writeRow(Constants.PROJECT_NAME_CSV_COL, Constants.PROJECT_VERSION_CSV_COL, Constants.PROJECT_ISSUE_TRACKER_CSV_COL, Constants.PROJECT_URL_CSV_COL, Constants.SEVERITY_CSV_COL, Constants.CLASS_NAME_CSV_COL, Constants.START_LINE_CSV_COL, Constants.END_LINE_CSV_COL, Constants.SOURCE_CODE_CSV_COL);

        D4JMethodExtractor d4JMethodExtractor = new D4JMethodExtractor();
        List<List<Method>> methods = d4JMethodExtractor.extractMethods(bugs);

        List<Method> buggyMethods = methods.get(0);
        List<Method> nonBuggyMethods = methods.get(1);
        List<Method> fixedMethods = methods.get(2);


        for (Method buggyMethod : buggyMethods) {
            D4JBug d4JBug = (D4JBug) buggyMethod.getRelatedBug();
            buggyMethodsCsvWriter.writeRow(d4JBug.getProjectName(), d4JBug.getProjectVersion(), d4JBug.getIssueTracker(), d4JBug.getUrl(), d4JBug.getSeverity(), buggyMethod.getFileName(), String.valueOf(buggyMethod.getStartLine()), String.valueOf(buggyMethod.getEndLine()), buggyMethod.getMethodString());
        }

        for (Method nonBuggyMethod : nonBuggyMethods) {
            D4JBug d4JBug = (D4JBug) nonBuggyMethod.getRelatedBug();
            nonBuggyMethodsCsvWriter.writeRow(d4JBug.getProjectName(), d4JBug.getProjectVersion(), d4JBug.getIssueTracker(), d4JBug.getUrl(), d4JBug.getSeverity(), nonBuggyMethod.getFileName(), String.valueOf(nonBuggyMethod.getStartLine()), String.valueOf(nonBuggyMethod.getEndLine()), nonBuggyMethod.getMethodString());
        }

        for (Method fixedMethod : fixedMethods) {
            D4JBug d4JBug = (D4JBug) fixedMethod.getRelatedBug();
            fixedMethodsCsvWriter.writeRow(d4JBug.getProjectName(), d4JBug.getProjectVersion(), d4JBug.getIssueTracker(), d4JBug.getUrl(), d4JBug.getSeverity(), fixedMethod.getFileName(), String.valueOf(fixedMethod.getStartLine()), String.valueOf(fixedMethod.getEndLine()), fixedMethod.getMethodString());
        }

        buggyMethodsCsvWriter.close();
        nonBuggyMethodsCsvWriter.close();
        fixedMethodsCsvWriter.close();
    }

    public void extractBugJarMethods() throws IOException {

        CsvReader csvReader = CsvReader.builder().build(Paths.get(Constants.BUGS_JAR_DATA_FILE));
        CloseableIterator<CsvRow> csvRow = csvReader.iterator();
        csvRow.next();
        List<Bug> bugs = csvReader.stream().map(row -> new BugsJarBug(row.getField(0), row.getField(1), row.getField(2))).collect(Collectors.toList());
        csvRow.close();

        CsvWriter buggyMethodsCsvWriter = CsvWriter.builder().build(Paths.get(Constants.DATA_DIR, Constants.BUGS_JAR_BUGGY_METHOD_FILE));
        CsvWriter nonBuggyMethodsCsvWriter = CsvWriter.builder().build(Paths.get(Constants.DATA_DIR, Constants.BUGS_JAR_NON_BUGGY_METHOD_FILE));
        CsvWriter fixedMethodsCsvWriter = CsvWriter.builder().build(Paths.get(Constants.DATA_DIR, Constants.BUGS_JAR_FIXED_METHOD_FILE));


        buggyMethodsCsvWriter.writeRow(Constants.PROJECT_NAME_CSV_COL, Constants.PROJECT_VERSION_CSV_COL, Constants.SEVERITY_CSV_COL, Constants.CLASS_NAME_CSV_COL, Constants.START_LINE_CSV_COL, Constants.END_LINE_CSV_COL, Constants.SOURCE_CODE_CSV_COL);
        nonBuggyMethodsCsvWriter.writeRow(Constants.PROJECT_NAME_CSV_COL, Constants.PROJECT_VERSION_CSV_COL, Constants.SEVERITY_CSV_COL, Constants.CLASS_NAME_CSV_COL, Constants.START_LINE_CSV_COL, Constants.END_LINE_CSV_COL, Constants.SOURCE_CODE_CSV_COL);
        fixedMethodsCsvWriter.writeRow(Constants.PROJECT_NAME_CSV_COL, Constants.PROJECT_VERSION_CSV_COL, Constants.PROJECT_ISSUE_TRACKER_CSV_COL, Constants.PROJECT_URL_CSV_COL, Constants.SEVERITY_CSV_COL, Constants.CLASS_NAME_CSV_COL, Constants.START_LINE_CSV_COL, Constants.END_LINE_CSV_COL, Constants.SOURCE_CODE_CSV_COL);

        BugJarMethodExtractor bugsJarMethodExtractor = new BugJarMethodExtractor();
        List<List<Method>> methods = bugsJarMethodExtractor.extractMethods(bugs);

        List<Method> buggyMethods = methods.get(0);
        List<Method> nonBuggyMethods = methods.get(1);
        List<Method> fixedMethods = methods.get(2);

        for (Method buggyMethod : buggyMethods) {
            BugsJarBug bugsjarBug = (BugsJarBug) buggyMethod.getRelatedBug();
            buggyMethodsCsvWriter.writeRow(bugsjarBug.getProjectName(), bugsjarBug.getProjectVersion(), bugsjarBug.getSeverity(), buggyMethod.getFileName(), String.valueOf(buggyMethod.getStartLine()), String.valueOf(buggyMethod.getEndLine()), buggyMethod.getMethodString());
        }

        for (Method nonBuggyMethod : nonBuggyMethods) {
            BugsJarBug bugsjarBug = (BugsJarBug) nonBuggyMethod.getRelatedBug();
            nonBuggyMethodsCsvWriter.writeRow(bugsjarBug.getProjectName(), bugsjarBug.getProjectVersion(), bugsjarBug.getSeverity(), nonBuggyMethod.getFileName(), String.valueOf(nonBuggyMethod.getStartLine()), String.valueOf(nonBuggyMethod.getEndLine()), nonBuggyMethod.getMethodString());
        }

        for (Method fixedMethod : fixedMethods) {
            BugsJarBug bugsjarBug = (BugsJarBug) fixedMethod.getRelatedBug();
            fixedMethodsCsvWriter.writeRow(bugsjarBug.getProjectName(), bugsjarBug.getProjectVersion(), bugsjarBug.getSeverity(), fixedMethod.getFileName(), String.valueOf(fixedMethod.getStartLine()), String.valueOf(fixedMethod.getEndLine()), fixedMethod.getMethodString());
        }

        buggyMethodsCsvWriter.close();
        nonBuggyMethodsCsvWriter.close();
        fixedMethodsCsvWriter.close();
    }
}
