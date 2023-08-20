package software.ehsan.severityprediction.metrics;

import software.ehsan.severityprediction.metrics.halstead.HalsteadMain;
import software.ehsan.severityprediction.metrics.halstead.HalsteadMetrics;

import java.text.DecimalFormat;

public class Halstead {

    public double calculateDifficulty(String sourceCode) {
        HalsteadMain halsteadMain = new HalsteadMain();
        HalsteadMetrics halsteadMetrics = halsteadMain.calculate(sourceCode);
        DecimalFormat decimalFormat = new DecimalFormat("##.00");
        double value = Double.parseDouble(decimalFormat.format(halsteadMetrics.getDifficulty()));
        return value;
    }

    public double calculateEffort(String sourceCode) {
        HalsteadMain halsteadMain = new HalsteadMain();
        HalsteadMetrics halsteadMetrics = halsteadMain.calculate(sourceCode);
        DecimalFormat decimalFormat = new DecimalFormat("##.00");
        double value = Double.parseDouble(decimalFormat.format(halsteadMetrics.getEffort()));
        return value;
    }

    public double calculateMaintenanceIndex(String sourceCode) {
        HalsteadMain halsteadMain = new HalsteadMain();
        McCabe mcCabe = new McCabe();
        SLOC sloc = new SLOC();
        HalsteadMetrics halsteadMetrics = halsteadMain.calculate(sourceCode);
        int mcCabeValue = mcCabe.calculateMcCabe(sourceCode);
        int slocValue = sloc.calculateSlocStandard(sourceCode);
        //check if it is correct for buggy vs non-buggy
        double value = (float) (171 - (5.2 * Math.log(halsteadMetrics.getVolume())) - (0.23 * mcCabeValue) - (16.2 * Math.log(slocValue)));
        DecimalFormat decimalFormat = new DecimalFormat("##.00");
        value = Double.parseDouble(decimalFormat.format(value));
        return value;
    }
}
