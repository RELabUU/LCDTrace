#Create a is_valid colomn. True if the trace link is valid.
def checkValidityTrace(issue_jira, issue_list_commit):
    # check if issue_key_jira occurs in the issue_list_commit
    for issue_commit in issue_list_commit:
        if (issue_jira == issue_commit):
            return(True)
        else:
            return(False)