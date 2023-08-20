package software.ehsan.severityprediction.method_extractor;

import software.ehsan.severityprediction.model.Bug;
import software.ehsan.severityprediction.model.Method;

import java.util.List;
import java.util.logging.Logger;

public interface MethodExtractor {
    Logger logger = Logger.getLogger(BugJarMethodExtractor.class.getName());
    List<List<Method>> extractMethods(List<Bug> bugs);
}
