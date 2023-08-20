from abc import abstractmethod, ABC
import json
from urllib.request import urlopen
from jira import JIRA
import requests
from bs4 import BeautifulSoup


class IssueScraper(ABC):

    @abstractmethod
    def get_severity(self, url, issue=None):
        pass


class ScrapeGoogleCodeIssues(IssueScraper):

    def get_severity(self, url, issue=None):
        json_url = urlopen(url)
        text = json.loads(json_url.read())
        return text["labels"]


class ScrapeJiraIssues(IssueScraper):

    def get_severity(self, url, issue=None):
        jira = JIRA(server=url)
        issue = jira.issue(issue)
        return issue.fields.priority


class ScrapeSourceForgeIssues(IssueScraper):

    def get_severity(self, url, issue=None):
        html_content = requests.get(url).text
        soup = BeautifulSoup(html_content, "lxml")
        candidates = (soup.findAll("div", attrs={"class": "grid-4"}))
        for i in range(len(candidates)):
            if "Priority" in candidates[i].text:
                return candidates[i].text.strip().split("\n")[1].strip()
