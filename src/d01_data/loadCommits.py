import re
import numpy as np
import json
import pandas as pd


def loadCommits(path):
    rawData_SVN_dataProcessing = open(path,"r", encoding="utf8")
    commit_processing_text = rawData_SVN_dataProcessing.read().strip()

    #Find every individual revision
    commits = re.findall(r"(?<=\n)(Revision+.*?)(?=\nRevision)", commit_processing_text, re.DOTALL)

    #Extract all individual data from commit
    email_list = []
    revision_list = []
    date_list = []
    log_list = []
    meta_data_list = []
    related_issue_key_list = []
    node_paths_list = []

    for commit in commits:
        #extract revision number
        revision = re.findall(r"(?<=Revision-number: )(.*)", commit)
        revision_string = ' '.join(revision)       
        revision_list.append(revision_string)
    
        #extract email
        email = re.findall(r"[\w\.-]+@[\w\.-]+", commit, re.DOTALL)
        email_string = ' '.join(email)
        email_list.append(email_string)
    
        #Extract node_paths
        node_paths = re.findall(r"(?<=Node-path: )(.*)(?=\n)", commit)
        node_path_string = ' '.join(node_paths)
        node_paths_list.append(node_path_string)
    
        #Extract meta-data
        meta_data = re.findall(r"(?:(?<=mx:metadata\nV [0-9]{1}\n)|(?<=mx:metadata\nV [0-9]{2}\n)|(?<=mx:metadata\nV [0-9]{3}\n)|(?<=mx:metadata\nV [0-9]{4}\n)|(?<=mx:metadata\nV [0-9]{5}\n))(.*?)(?=\nK)", commit, re.DOTALL)
        meta_data_string = ' '.join(meta_data)
        if(meta_data_string != ''):
            meta_data_dict = json.loads(meta_data_string)
        else:
            meta_data_dict = np.NaN
        meta_data_list.append(meta_data_dict)
    
        #extract log
        log = re.findall(r"(?:(?<=svn:log\nV [0-9]{1})|(?<=svn:log\nV [0-9]{2})|(?<=svn:log\nV [0-9]{3}))\n(.*?)\nPROPS-END", commit, re.DOTALL)
        log_string = ''.join(log)
        log_list.append(log_string)
    
        #Extract the issue key from the logs from the projects LRN, AFM, MA, AFI, EM, OE, and EM
        jira_issue = re.findall(r"LRN+.[0-9]+|AFM+.[0-9]+|MA+.[0-9]+|AFI+.[0-9]+|EM+.[0-9]+|OE+.[0-9]+|EM+.[0-9]+", log_string)
        if (jira_issue):
            cleaned_issues = []
            for item in jira_issue:
                cleaned_item = item.replace(" ", "-")
                cleaned_issues.append(cleaned_item)
            related_issue_key_list.append(cleaned_issues)
        if not (jira_issue):
            related_issue_key_list.append(np.NaN)
    
        #extract date
        date = re.findall(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{6}Z", commit)
        date_string = ''.join(date)
        date_list.append(date[0])

    #Extract the usefull data of the meta data
    modeler_version_list = []
    related_stories_list = []
    branch_name_list = []
    model_changes_list = []
    impacted_unit_names_list = []
    impacted_modules_list = []
    impacted_module_types_list = []

    for meta_data_item in meta_data_list:
        #Extract the modeler version
        try:
            modeler_version_list.append(meta_data_item['ModelerVersion'])
        except Exception as e:
            modeler_version_list.append(np.NaN)
        
        #Extract the related stories
        try:
            related_stories_list.append(meta_data_item['RelatedStories'])
        except Exception as e:
            related_stories_list.append(np.NaN)
        
        #Extract the branch_name
        try:
            branch_name_list.append(meta_data_item['BranchName'])
        except Exception as e:
            branch_name_list.append(np.NaN)

        #Extract the meta_data
        try:
            #parse json array to python list
            meta_data_dump = json.dumps(meta_data_item['ModelChanges'])
   
            meta_data_dump_list = json.loads(meta_data_dump)
            model_changes_string = ""
            modules_string = ""
            module_types_string = ""
       
            for meta_data_dump in meta_data_dump_list:
                model_changes_string = model_changes_string + " " + meta_data_dump['UnitName']
                modules_string = modules_string + " " + meta_data_dump['Module']
                module_types_string = module_types_string + " " + meta_data_dump['UnitType']
            
            impacted_unit_names_list.append(model_changes_string)
            impacted_modules_list.append(modules_string)
            impacted_module_types_list.append(module_types_string)
        
            model_changes_list.append(meta_data_dump_list)    
        except Exception as e:
            impacted_unit_names_list.append(np.NaN)
            impacted_modules_list.append(np.NaN)
            model_changes_list.append(np.NaN) 
            impacted_module_types_list.append(np.NaN)
        

    #Extract all unit_names from the model_changes_list
        

        
    #Create dictionary
    commit_dictionary = {'revision': revision_list, 
                        'email':email_list,
                        'date':date_list, 
                        'log': log_list,
                        'node_paths': node_paths_list,
                        'related_issue_key': related_issue_key_list,
                        'modeler_version': modeler_version_list,
                        'branch_name': branch_name_list,
                        'related_stories': related_stories_list,
                        'model_changes_list': model_changes_list,
                        'impacted_unit_names': impacted_unit_names_list,
                        'impacted_modules' : impacted_modules_list,
                        'impacted_module_types': impacted_module_types_list
                        }

    #Transform the list to a dataframe
    commit_df = pd.DataFrame(commit_dictionary)
    return(commit_df)