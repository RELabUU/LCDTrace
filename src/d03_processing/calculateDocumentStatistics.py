#Number_unique_terms_in_document1
from collections import Counter
import numpy as np

def calculateUniqueWordCount(termList):
    try:
        uniqueWordCount = len(list(Counter(termList).keys()))
    except:
        uniqueWordCount = np.nan
    return(uniqueWordCount)

def calculateTotalWordCount(termList):
    try:
        totalWordCount = len(termList)
    except:
        totalWordCount = np.nan
    return(totalWordCount)

def calculateOverlapBetweenDocuments(termList1, termList2, comparisonList):
    if(isinstance(termList1, list) & isinstance(termList2, list)):
        set1 = set(termList1)
        set2 = set(termList2)
        overlap = set1 & set2
        union = set1 | set2
  
        #Compare the overlap to list1
        if(comparisonList == 'list1'):
            percentageOverlap = float(len(overlap)) / len(set1) * 100
        #Compare the overlap to list2
        elif(comparisonList == 'list2'):
            percentageOverlap = float(len(overlap)) / len(set2) * 100
        elif(comparisonList == 'union'):
            percentageOverlap = float(len(overlap)) / len(union) * 100
        return(percentageOverlap)
    else:
        return(np.nan)