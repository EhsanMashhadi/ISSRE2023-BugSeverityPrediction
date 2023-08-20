package software.ehsan.severityprediction.metrics;/*
Adopted and corrected from: https://github.com/rodhilton/jasome

Changes from the original implementation:
1) ForEach was missing we added
2) We subtracted 1 because minimum should be zero
 */

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.BodyDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.LambdaExpr;
import com.github.javaparser.ast.stmt.*;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;

public class NestedBlockDepth {

    public int calculateNBD(String code) {

        BodyDeclaration methodDeclaration = JavaParser.parseBodyDeclaration(code);
        List<BlockStmt> blocks = methodDeclaration.getNodesByType(BlockStmt.class);
        List<SwitchEntryStmt> switchEntries = methodDeclaration.getNodesByType(SwitchEntryStmt.class);

        List<Node> allNestedBlocks = new ArrayList<>();
        allNestedBlocks.addAll(blocks);
        allNestedBlocks.addAll(switchEntries);

        OptionalInt maxDepth = allNestedBlocks.parallelStream().mapToInt(block -> {
            //figure out this block's depth and return it
            Node theNode = block;
            int i = 1;
            while (theNode != methodDeclaration) {
                if (theNode instanceof IfStmt ||
                        theNode instanceof SwitchEntryStmt ||
                        theNode instanceof SwitchStmt ||
                        theNode instanceof TryStmt ||
                        theNode instanceof ForStmt ||
                        theNode instanceof ForeachStmt ||
                        theNode instanceof WhileStmt ||
                        theNode instanceof DoStmt ||
                        theNode instanceof LambdaExpr ||
                        theNode instanceof ClassOrInterfaceDeclaration ||
                        theNode instanceof MethodDeclaration ||
                        theNode instanceof SynchronizedStmt) {
                    //Javaparser has an interesting relationship that shows up here.. basically if you have something like an
                    //if statement, even though that "nests" 1 level, the block statement itself is a separate thing
                    //with the if statement as a parent, which means that we'd count it two.  A few other classes nest like this
                    //so we have to only increase the counter when the node we're looking at isn't one of them.  Thus we
                    //essentially whitelist the kind of statements that DO increase nesting
                    i++;
                }
                if (theNode.getParentNode().isPresent()) {
                    theNode = theNode.getParentNode().get();
                } else {
                    break;
                }
            }
            return i;
        }).max();

        return maxDepth.orElse(1) - 1;
    }

}
