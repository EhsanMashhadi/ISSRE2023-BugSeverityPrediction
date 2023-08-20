package software.ehsan.severityprediction.metrics;


import raykernel.apps.readability.eval.Main;

import java.text.DecimalFormat;

public class Readability {

    public double calculateReadability(String sourceCode) {
        DecimalFormat decimalFormat = new DecimalFormat("##.00");
        double value = Main.getReadability(sourceCode);
        value = Double.parseDouble(decimalFormat.format(value));
        return value;
    }
}
