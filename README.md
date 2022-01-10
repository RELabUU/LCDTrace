# LCDTrace
**Keywords:** Requirements Traceability - Trace Link Recovery - Model-Driven Development - Low-Code Development - Machine Learning

## About
This repository provides the code used to produce the results of Rasiman, Dalpiaz & España (2022). It takes a set of JIRA issues, and SVN commits as input. The program then cleans and preprocesses these. The JIRA id (label) is then taken from the SVN commit logs and appended to the commit data using REGEX. The Cartesian product of the JIRA issues and SVN commits is then produced, with each element being a candidate trace. For each candidate trace, a set of features is computed. These features are then utilized as input data for 12 different models (classification algorithm x rebalancing strategy). Finally, these models are evaluated.

**Input:**
* JIRA Issues (.cvs .xlsx)
    A .csv or .xlsx dump of a JIRA project.
* Mendix SVN Dump (.txt)

**Output:**
* The accuracy, precision, recall, f1-score, f2-score, f0.5-score of the fitted models (on the training set) used to predict te labels of the test set
* the feature importances of the fitted models

# Dependencies
* [Python](https://www.python.org/) (>= 3.9.4)
* [Jupyter Notebook](https://jupyter.org/) (>=  6.2.0)
* [Pandas](https://pandas.pydata.org/) (>=  1.2.4)
* [NumPy](https://numpy.org/) (>=  1.20.1)
* [SciPy](https://scipy.org/) (>=  1.6.2)
* [NLTK](https://www.nltk.org/) (>=  3.5)
* [Scikit-Learn](https://scikit-learn.org) (>=  0.24.1)

# Directory Structure
The project follows the structure laid out by [Data Science for Social Good](https://github.com/dssg/hitchhikers-guide). Below is a modified of their system as used in this project
```
├── data  
│   ├── 01_raw                              <- Imutable input data
│   │   ├── jira_example.csv                <- example  of raw JIRA input
│   │   └── svn_example.txt                 <- example  of svn input
│   ├── 02_intermediate                     <- Cleaned version of raw
│   └── 03_processed                        <- The data used for modelling 
├── notebooks                               <- Jupyter notebooks
│   └── lcd_trace.ipynb                     <- Notebook to evaluate trace classification models
├── results                                 <- The evaluation results produced by the notebook
│   ├── 02_non_normalised_results           <- The evaluation for non-normalised feature input data
│   │   ├── light_gbm              
│   │   ├── random_forests
│   │   └── xg_boost  
│   ├── 03_normalised_results               <- The evaluation for normalised feature input data
│   │   ├── light_gbm              
│   │   ├── random_forests
│   │   └── xg_boost  
│   └── 04_feature_imporance_results        <- The feature importances for the different models
│   │   ├── light_gbm              
│   │   ├── random_forests
│   │   └── xg_boost  
├── src                                     <- Source code for use in this project.
│   ├── d00_utils                           <- Methods used across the project
│   │   └── calculateTimeDifference.py      <- Method to give time difference in minutes and seconds (string)
│   ├── d01_data                            <- Scripts to reading and writing data
│   │   └── loadCommits.py                  <- Method to load .txt to a pandas dataframe
│   ├── d02_intermediate                    <- Scripts to transform data from raw to intermediate
│   │   ├── cleanCommitData.py              <- Methods to clean SVN commit data
│   │   └── cleanJiraData.py                <- Methods to clean JIRA data
│   ├── d03_processing                      <- Scripts to turn intermediate data into modelling input
│   │   ├── calculateCosineSimilarity.py    <- Methods to calculate cosine similarity between 2 documents
│   │   ├── calculateDocumentStatistics.py  <- Methods to compute document statistics features
│   │   ├── calculateQueryQuality.py        <- Methods to compute query quality features
│   │   ├── calculateTimeDif.py             <- Method to calculate the time difference between 2 dates in seconds
│   │   ├── checkFullnameEqualsEmail.py     <- Method to check if the full_name is part of the email address
│   │   ├── checkValidityTrace.py           <- Method to check if a trace is valid or invalid
│   │   ├── createCorpusFromDocumentList.py <- Method to create a corpus
│   │   ├── createFittedTF_IDF.py           <- Method to fit a tf_idf on a document
│   │   └── normalize_data.py               <- Method to normalize data
│   └── d04_model_evaluation                <- Scripts that analyse model performance and model selection
│   │   └── model_evaluation.py             <- Methods to produce the evaluation metrics
└── readme.md                 
```

# Running the project
1. Start with the project directory as the current directory. Execute Jupyter Notebook by executing the following code in your CLI:
```
python -m notebook
```
2. Navigate to ```notebooks``` and execetute ```lcd_trace.ipynb```.
3. In the second cell block provide the file path to the JIRA set and the SVN commit set. An example is given underneath using example sets.
```
#Import raw JIRA data as a pandas dataframe
jira_df_raw = pd.read_csv('../data/01_raw/jira_example.csv')

#Import raw svn data as a pandas dataframe
svn_df_raw = loadCommits('../data/01_raw/svn_example.txt')
```
4. Navigate to ```Cell``` and then ```Run all cells```


# Publication
Rasiman, R.S., Dalpiaz, F., & España, S. (2022). How Effective Is Automated Trace Link Recovery in Model-Driven Development? In *International Working Conference on Requirements Engineering: Foundation for Software Quality*. Springer, Nature.
