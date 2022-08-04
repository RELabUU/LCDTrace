import numpy as np

def checkFullnameEqualsEmail(fullname, email):
    if(isinstance(fullname, str)):    
        name_in_mail = email.split("@")[0]
        name_in_mail_clean = name_in_mail.replace('.', ' ')
        fullname_lower = fullname.lower()
    
        if(fullname_lower == name_in_mail_clean):    
            return(1)
        else:
            return(0)
    else:
        return(np.nan)