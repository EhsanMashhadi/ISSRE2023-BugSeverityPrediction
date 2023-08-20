/*

Total fanout is very accurate.

For fanout we only miss two categories, which are rare:

1) method(int, float) and method(flot, int) they are the same to us..

2) class1.method() and class2.method() are same to us..

This does not incur any false positive but makes the process extremely faster than using
symbol analysis
 */

package software.ehsan.severityprediction.metrics;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.body.BodyDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;

import java.util.List;

public class StructuralDependency {

    private int fanOut;

    public StructuralDependency() {
        this.fanOut = -1;
    }

    public void calculateFanOut(String code) {
        BodyDeclaration methodDeclaration = JavaParser.parseBodyDeclaration(code);
        List<MethodCallExpr> methodCallExprsList = methodDeclaration.findAll(MethodCallExpr.class);
        setFanOut(methodCallExprsList.size());
    }

    public int getFanOut() {
        return fanOut;
    }

    public void setFanOut(int fanOut) {
        this.fanOut = fanOut;
    }


}