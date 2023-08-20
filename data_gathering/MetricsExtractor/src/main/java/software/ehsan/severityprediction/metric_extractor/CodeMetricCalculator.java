package software.ehsan.severityprediction.metric_extractor;

import software.ehsan.severityprediction.model.Method;
import software.ehsan.severityprediction.metrics.*;

import java.util.List;

public class CodeMetricCalculator {

    protected void setMethods(List<Method> methods) {
        for (Method method : methods) {
            String sourceCode = method.getMethodString();
            method.setCodeMetrics(extractMetrics(sourceCode));
        }
    }

    private CodeMetrics extractMetrics(String sourceCode) {

        SLOC slocCalculator = new SLOC();
        ProxyIndentation indentationComplexity = new ProxyIndentation();
        McCabe mcCabeCalculator = new McCabe();
        NestedBlockDepth nestedBlockDepthCalculator = new NestedBlockDepth();
        McClure mcClureCalculator = new McClure();

        int sloc = slocCalculator.calculateSlocStandard(sourceCode);
        double indentation = indentationComplexity.calculateIndentation(sourceCode);
        double mcCabe = mcCabeCalculator.calculateMcCabe(sourceCode);
        double nestedBlockDepth = nestedBlockDepthCalculator.calculateNBD(sourceCode);
        mcClureCalculator.calculateMcClure(sourceCode);
        double mcClure = mcClureCalculator.getMCLC();
        Halstead halstead = new Halstead();
        double difficulty = halstead.calculateDifficulty(sourceCode);
        double effort = halstead.calculateEffort(sourceCode);
        double maintenanceIndex = halstead.calculateMaintenanceIndex(sourceCode);

        StructuralDependency structuralDependency = new StructuralDependency();
        structuralDependency.calculateFanOut(sourceCode);
        double totalFanOut = structuralDependency.getFanOut();

        structuralDependency = new StructuralDependency();
        structuralDependency.calculateFanOut(sourceCode);

        Readability readabilityCalculator = new Readability();
        double readability = readabilityCalculator.calculateReadability(sourceCode);

        CodeMetrics codeMetrics = new CodeMetrics(sloc, indentation, mcCabe, nestedBlockDepth, mcClure, difficulty, maintenanceIndex, readability, totalFanOut, effort);
        return codeMetrics;
    }
}
