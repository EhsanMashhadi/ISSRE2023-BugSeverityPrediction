package software.ehsan.severityprediction.metrics;/*
Java based Implementation of,
Reading Beside the Lines: Indentation as a Proxy for Complexity Metrics Abram Hindle et al., ICPC paper.
Section 5 states that logical indentation does not matter, so we only care about raw indentation.
 */

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.body.BodyDeclaration;
import com.github.javaparser.ast.comments.Comment;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.text.DecimalFormat;
import java.util.List;

public class ProxyIndentation {

    DecimalFormat decimalFormat = new DecimalFormat("##.00");
    private DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();

    public double calculateIndentation(String code) {
        //Meeting: Excluding the comments (possible)
        descriptiveStatistics.clear();
        BodyDeclaration cu = JavaParser.parseBodyDeclaration(code);
        List<Comment> comments = cu.getAllContainedComments();

        for (Comment comment : comments) {
            if (comment.isBlockComment() || comment.isJavadocComment()) {
                code = code.replace(comment.toString().trim(), "");
            }
        }

        String[] lines = code.split("\r\n|\r|\n");
        int spaceCount;
        boolean isBlankLine;

        for (String line : lines) {
            // adopted and modified from https://stackoverflow.com/questions/9655753/how-to-count-the-spaces-in-a-java-string
            if (line.trim().equals("") || line.trim().startsWith("//")) {
                continue;
            }
            spaceCount = 0;
            isBlankLine = true;
            for (char c : line.toCharArray()) {
                if (c == ' ' || c == '\t') {
                    if (c == ' ') {
                        spaceCount++;
                    } else {
                        spaceCount += 8; // tab is equivalent to 8 spaces, section 3.1, last paragraph
                    }
                } else {
                    isBlankLine = false;// it was not a blank line
                    break;
                }
            }

            if (!isBlankLine) {
                this.descriptiveStatistics.addValue(spaceCount);
            }
        }
        // because from figure 1 in the paper, and communication with Abram,
        //he calculated population standard deviation..
        double standardDeviation = Double.parseDouble(decimalFormat.format(Math.sqrt(descriptiveStatistics.getPopulationVariance())));
        return standardDeviation;
    }
}
