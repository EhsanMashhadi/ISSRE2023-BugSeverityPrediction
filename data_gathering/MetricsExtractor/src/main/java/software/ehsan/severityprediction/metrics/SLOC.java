package software.ehsan.severityprediction.metrics;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.body.BodyDeclaration;
import com.github.javaparser.ast.comments.Comment;

import java.util.List;

public class SLOC {

    public int calculateSlocStandard(String code) {

        BodyDeclaration cu = JavaParser.parseBodyDeclaration(code);
        List<Comment> comments = cu.getAllContainedComments();

        for (Comment comment : comments) {
            if (comment.isBlockComment() || comment.isJavadocComment()) {
                code = code.replace(comment.toString().trim(), "");
            }
        }

        int total = calculateSLOCAsItIs(code);
        // calculate total blank lines
        int blank = 0;
        String[] lines = code.split("\r\n|\r|\n");
        for (String line : lines) {
            if (line.trim().equals("") || line.trim().startsWith("//")) {
                blank++;
            }
        }
        return total - blank;
    }

    private int calculateSLOCAsItIs(String code) {
        String[] lines = code.split("\r\n|\r|\n");
        return lines.length;
    }
}