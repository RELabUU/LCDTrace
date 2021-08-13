from sklearn.feature_extraction.text import TfidfTransformer 

def createFittedTF_IDF(cv, corpus):
    # Generate the word counts for the corpus
    word_count_vector = cv.fit_transform(corpus)
    tfidf_transformer = TfidfTransformer(smooth_idf = True, use_idf = True) 
    fittedTF_IDF = tfidf_transformer.fit(word_count_vector)
    return(fittedTF_IDF)