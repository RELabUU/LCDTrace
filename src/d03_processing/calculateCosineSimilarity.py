from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer 
from scipy import spatial
import pandas as pd


def calc_vector_representation(document, corpus):
    #instantiate CountVectorizer() 
    cv = CountVectorizer()
    
    # Generate the word counts for the corpus
    word_count_vector = cv.fit_transform(corpus)
    tfidf_transformer = TfidfTransformer(smooth_idf = True, use_idf = True) 
    tfidf_transformer.fit(word_count_vector)
    
    #Transform document type to a string
    documentString = document
    
    #Calculate the Term Frequency of the document
    inputDocs = [documentString] 

    # count matrix 
    count_vector = cv.transform(inputDocs) 
 
    #tf-idf scores 
    tf_idf_vector = tfidf_transformer.transform(count_vector)

    feature_names = cv.get_feature_names() 
 
    #get tfidf vector for first document 
    document_vector=tf_idf_vector[0] 
 
    #print the scores 
    
    # place tf-idf values in a pandas data frame 
    df = pd.DataFrame(document_vector.T.todense(), index=feature_names, columns=["tfidf"]) 
    df.sort_values(by=["tfidf"],ascending=False)

    return(document_vector.T.todense())

def calculateCosineSimilarity(document1, document2, corpus):
    #Transform document to string type
    document1String = ' '.join(document1)
    document2String = ' '.join(document2)

    vector1 = calc_vector_representation(document1String, corpus)
    vector2 = calc_vector_representation(document2String, corpus)
    
    #The cosine similarity. Produces NaN if no terms are found in the corpus.
    result = 1 - spatial.distance.cosine(vector1, vector2)
    
    return(result)