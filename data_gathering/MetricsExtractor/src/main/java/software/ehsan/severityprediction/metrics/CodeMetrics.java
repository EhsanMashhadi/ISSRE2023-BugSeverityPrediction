package software.ehsan.severityprediction.metrics;

public class CodeMetrics {

    double sloc;
    double proxyIndentation;
    double mcCabe;
    double nestedBlockDepth;
    double mcClure;
    double difficulty;
    double maintainabilityIndex;
    double readability;
    double fanOut;
    double effort;

    public CodeMetrics(double sloc, double indentation, double mcCabe, double nestedBlockDepth, double mcClure, double difficulty, double maintenanceIndex, double readability, double totalFanOut, double effort) {
        this.sloc = sloc;
        this.proxyIndentation = indentation;
        this.mcCabe = mcCabe;
        this.nestedBlockDepth = nestedBlockDepth;
        this.mcClure = mcClure;
        this.difficulty = difficulty;
        this.maintainabilityIndex = maintenanceIndex;
        this.readability = readability;
        this.fanOut = totalFanOut;
        this.effort = effort;
    }

    public double getSloc() {
        return sloc;
    }

    public double getProxyIndentation() {
        return proxyIndentation;
    }

    public double getMcCabe() {
        return mcCabe;
    }

    public double getNestedBlockDepth() {
        return nestedBlockDepth;
    }

    public double getMcClure() {
        return mcClure;
    }

    public double getDifficulty() {
        return difficulty;
    }

    public double getMaintainabilityIndex() {
        return maintainabilityIndex;
    }

    public double getReadability() {
        return readability;
    }

    public double getFanOut() {
        return fanOut;
    }

    public double getEffort() {
        return effort;
    }
}
