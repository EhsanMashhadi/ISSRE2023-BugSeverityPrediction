package software.ehsan.severityprediction.metrics;/*
McCabe Calculation: Empirical analysis of the relationship between CC and metrics.SLOC in a large corpus of Java methods

Adopted from:
https://github.com/rodhilton/jasome
https://www.guru99.com/cyclomatic-complexity.html#4
https://perso.ensta-paris.fr/~diam/java/online/notes-java/principles_and_practices/complexity/complexity-java-method.html
 */

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.body.BodyDeclaration;
import com.github.javaparser.ast.expr.ConditionalExpr;
import com.github.javaparser.ast.stmt.*;

import java.util.List;

public class McCabe {

    public int calculateMcCabe(String code) {

        BodyDeclaration methodDeclaration = JavaParser.parseBodyDeclaration(code);
        List<IfStmt> ifStmts = methodDeclaration.findAll(IfStmt.class);
        List<ForStmt> forStmts = methodDeclaration.findAll(ForStmt.class);
        List<WhileStmt> whileStmts = methodDeclaration.findAll(WhileStmt.class);
        List<DoStmt> doStmts = methodDeclaration.findAll(DoStmt.class);
        List<CatchClause> catchClauses = methodDeclaration.findAll(CatchClause.class);
        List<ConditionalExpr> ternaryExpressions = methodDeclaration.findAll(ConditionalExpr.class);
        List<ForeachStmt> forEachStmts = methodDeclaration.findAll(ForeachStmt.class);
        List<SwitchEntryStmt> switchEntryStmts = methodDeclaration.findAll(SwitchEntryStmt.class);

        List<BreakStmt> breakStmts = methodDeclaration.findAll(BreakStmt.class);
        List<ContinueStmt> continueStmts = methodDeclaration.findAll(ContinueStmt.class);
        List<ThrowStmt> throwStmts = methodDeclaration.findAll(ThrowStmt.class);

        int total = ifStmts.size() + forStmts.size() + forEachStmts.size() + whileStmts.size() + doStmts.size() +
                switchEntryStmts.size() + catchClauses.size() + ternaryExpressions.size() + breakStmts.size() +
                continueStmts.size() + throwStmts.size() + 1;

        return total;

    }
}

