package software.ehsan.severityprediction.model;

public class D4JBug extends Bug {

    private String projectVersion;
    private String issueTracker;
    private String url;

    public D4JBug(String projectName, String projectVersion, String issueTracker, String url, String severity) {
        super(projectName, severity);
        this.projectVersion = projectVersion;
        this.issueTracker = issueTracker;
        this.url = url;
    }

    public String getProjectVersion() {
        return projectVersion;
    }

    public String getIssueTracker() {
        return issueTracker;
    }

    public String getUrl() {
        return url;
    }
}
