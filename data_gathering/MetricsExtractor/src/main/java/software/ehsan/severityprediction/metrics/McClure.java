/*
Adopted and corrected from: https://github.com/rodhilton/jasome

for numberOfcomparisons:
we take all the expressions from if, conditional (?:),
for, while, do,  switch entries (default should no be there).
NOT CONSIDERING CATCH CLAUSES AFTER DISCUSSING WITH REID

for control variables: anything that is not a constant, even a method that comes with a
value in runtime.

WEAKNESSES: In Swtich selector we could not accurately count the number of comparisons..
We are either correct, or one less.. but never more. So no false positive
 */

package software.ehsan.severityprediction.metrics;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.body.BodyDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.*;
import software.ehsan.severityprediction.utils.Util;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class McClure {

    public Set<String> namesNVR;
    int NCOMP;
    int NVAR;
    int MCLC;
    int maxCompareInOneExpression;
    int countCompare;

    public McClure() {

        NCOMP = -1; //Total number of compare;
        NVAR = -1; // number of control variable;
        MCLC = -1;
        countCompare = 0;
        maxCompareInOneExpression = 0;
        namesNVR = new HashSet<>();

    }

    public int getMaxCompareInOneExpression() {
        return maxCompareInOneExpression;
    }

    public void setMaxCompareInOneExpression(int maxCompareInOneExpression) {
        this.maxCompareInOneExpression = maxCompareInOneExpression;
    }

    public int getNCOMP() {
        return NCOMP;
    }

    public void setNCOMP(int NCOMP) {
        this.NCOMP = NCOMP;
    }

    public int getNVAR() {
        return NVAR;
    }

    public void setNVAR(int NVAR) {
        this.NVAR = NVAR;
    }

    public int getMCLC() {
        return MCLC;
    }

    public void setMCLC(int MCLC) {
        this.MCLC = MCLC;
    }


    public void calculateMcClure(String code) {

        BodyDeclaration methodDeclaration = JavaParser.parseBodyDeclaration(code);
        // List<BinaryExpr> comparisons = methodDeclaration.findAll(BinaryExpr.class);
        List<Expression> conditionalExprs = new ArrayList<>();

        conditionalExprs.addAll(
                methodDeclaration.findAll(IfStmt.class).stream().map(IfStmt::getCondition).collect(Collectors.toList())
        );

        conditionalExprs.addAll(
                methodDeclaration.findAll(ConditionalExpr.class).stream().map(ConditionalExpr::getCondition).collect(Collectors.toList())
        );

        conditionalExprs.addAll(
                methodDeclaration.findAll(WhileStmt.class).stream().map(WhileStmt::getCondition).collect(Collectors.toList())
        );

        conditionalExprs.addAll(
                methodDeclaration.findAll(DoStmt.class).stream().map(DoStmt::getCondition).collect(Collectors.toList())
        );

        conditionalExprs.addAll(
                methodDeclaration.findAll(ForStmt.class)
                        .stream()
                        .map(ForStmt::getCompare)
                        .flatMap(o -> o.isPresent() ? Stream.of(o.get()) : Stream.empty())
                        .collect(Collectors.toList())
        );

        conditionalExprs.addAll(
                methodDeclaration.findAll(SwitchStmt.class).stream().map(SwitchStmt::getSelector).collect(Collectors.toList())
        );


        for (Expression expr : conditionalExprs) {
            calculateNVR(expr);
            calculateNCOMP(expr);
        }


        setNVAR(namesNVR.size());
        setNCOMP(countCompare);
        doCorrection(methodDeclaration);
        setMCLC(getNVAR() + getNCOMP());
        namesNVR = new HashSet<>(); // so that we clear the memory


    }

    public void doCorrection(BodyDeclaration methodDeclaration) {
        // we need to delete numberOfSwitchSelector from # of comparisons
        // because we had to count them for our way of implementatiom

        int numberOfSwitchSelector = methodDeclaration.findAll(SwitchStmt.class).size();
        // we need to add numberOfSwitchEntries in # of comparisons
        int switchCases = 0;
        List<SwitchEntryStmt> switchEntryStmts = methodDeclaration.findAll(SwitchEntryStmt.class);

        for (SwitchEntryStmt stmt : switchEntryStmts) {
            if (stmt.getLabel().isPresent()) {
                //ignore default
                switchCases++;
            }
        }

        setNCOMP((getNCOMP() - numberOfSwitchSelector) + switchCases);

        if (getNCOMP() == 0) {
            //just in case because of our possible inaccuracy with swtich selector
            setMaxCompareInOneExpression(0);
        }
    }


    public void calculateNCOMP(Expression expression) {

        //these are all conditional expression, so there is at least one compare
        int sum = 1;
        sum += Util.countOccurence(expression.toString(), "&&");
        sum += Util.countOccurence(expression.toString(), "||");

        if (maxCompareInOneExpression < sum) {
            maxCompareInOneExpression = sum;
        }
        countCompare += sum;

    }

    public void calculateNVR(Expression expression) {

        if (expression.getClass().getSimpleName().equals("EnclosedExpr")) {
            processEnclosedExpr((EnclosedExpr) expression);

        }
        if (expression.getClass().getSimpleName().equals("FieldAccessExpr")) {
            processFieldAccessExpr((FieldAccessExpr) expression);
        }

        if (expression.getClass().getSimpleName().equals("NameExpr")) {
            processNameExpr((NameExpr) expression);
        }

        if (expression.getClass().getSimpleName().equals("BinaryExpr")) {
            processBinaryExpr((BinaryExpr) expression);
        }

        if (expression.getClass().getSimpleName().equals("UnaryExpr")) {
            processUnaryExpr((UnaryExpr) expression);
        }

        if (expression.getClass().getSimpleName().equals("InstanceOfExpr")) {
            processInstanceOfExpr((InstanceOfExpr) expression);
        }

        if (expression.getClass().getSimpleName().equals("LambdaExpr")) {
            processLambdaExpr((LambdaExpr) expression);
        }


        if (expression.getClass().getSimpleName().equals("MethodCallExpr")) {
            processMethodCallExpr((MethodCallExpr) expression);
        }

    }

    public void processMethodCallExpr(MethodCallExpr expression) {

        // A method call itself should be treated like a control variable,
        //because it can return different values: not constant
        namesNVR.add(expression.toString());

        for (Expression e : expression.getArguments()) {
            calculateNVR((Expression) e);

        }
    }

    public void processFieldAccessExpr(FieldAccessExpr expression) {

        //check if the name is not a constant
        if (!Util.isAllUpperCase(expression.getName().toString())) {
            namesNVR.add(expression.toString());
        }

    }

    public void processNameExpr(NameExpr expression) {

        if (!Util.isAllUpperCase(expression.getName().toString())) {
            namesNVR.add(expression.toString());
        }

    }

    public void processBinaryExpr(BinaryExpr expression) {

        // System.out.println(expression+"\n"+expression.getLeft()+"\t"+expression.getRight());
        calculateNVR(expression.getLeft());
        calculateNVR(expression.getRight());
    }

    public void processUnaryExpr(UnaryExpr expression) {
        calculateNVR(expression.getExpression());
    }

    public void processInstanceOfExpr(InstanceOfExpr expression) {
        //System.out.println(expression+"\t"+expression.getType());
        //comparison.add(expression.toString());
        calculateNVR(expression.getExpression());
    }

    public void processLambdaExpr(LambdaExpr expression) {
        try {
            for (Parameter parameter : expression.getParameters()) {
                calculateNVR(JavaParser.parseExpression(parameter.toString()));
            }
            calculateNVR(expression.getExpressionBody().get());

        } catch (Exception e) {
            System.out.println("Problem with lambda expression");
        }
    }

    public void processEnclosedExpr(EnclosedExpr expression) {

        calculateNVR(expression.getInner());

    }

}