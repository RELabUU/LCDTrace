# About
This repository provides the code used to produce the results of Van Oosten, Rasiman, Dalpiaz & Hurkmans (2022). It takes a set of JIRA issues, and SVN commits as input. The program then cleans and preprocesses these. The JIRA id (label) is then taken from the SVN commit logs and appended to the commit data using REGEX. The Cartesian product of the JIRA issues and SVN commits is then produced, with each element being a candidate trace. For each candidate trace, a set of features is computed. These features are then utilized as input data for 12 different models (classification algorithm x rebalancing strategy). Finally, these models are evaluated. This is then followed by a similar approach for non-MDD specific features and feature subsets (by an automated feature selection algorithm).

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
* [Skikit-posthocs](https://github.com/maximtrp/scikit-posthocs) (>= 0.7.0)
* [mrmr_selection](https://github.com/smazzanti/mrmr) (>=0.2.5)

# Directory Structure
The project follows the structure laid out by [Data Science for Social Good](https://github.com/dssg/hitchhikers-guide). Below is a modified of their system as used in this project
```
├── data  
│   ├── 01_raw                                      <- Imutable input data
│   │   ├── jira_example.csv                        <- Example of raw JIRA input
│   │   └── svn_example.txt                         <- Example of svn input
│   ├── 02_intermediate                             <- Cleaned version of raw
│   └── 03_processed                                <- The data used for modelling 
├── notebooks                                       <- Jupyter notebooks
│   ├── lcd_trace.ipynb                             <- Notebook to engineer the features to be used by the models
│   │   lcd_trace_no_lc_features.ipynb              <- Notebook to engineer the non-MDD specific features to be used by the models
│   │   evaluation_notebook.ipynb                   <- Notebook to evaluate trace classification models
│   │   feature_selection.ipynb                     <- Notebook to apply feature selection for three subset sizes
│   └── results processing
│       ├── calculating_feature_importance.ipynb    <- Notebook to calculate feature importance (families) and create boxplots
│       ├── Friedman-Nemenyi.ipynb                  <- Notebook to perform the statistical tests
│       └── process_results.ipynb                   <- Notebook to process the raw results 
├── results                                         <- The evaluation results produced by the notebook
│   ├── 01_Trace link Feature Data                  <- Example of populated feature files
│   ├── 02_non_normalised_results                   <- The evaluation for non-normalised feature input data
│   │   ├── light_gbm              
│   │   ├── random_forests
│   │   └── xg_boost  
│   ├── 04_feature_imporance_results                <- The feature importances for the different models
│   │   ├── light_gbm              
│   │   ├── random_forests
│   │   └── xg_boost
│   ├── 05_Feature selection subsets                <- Examples of proccessed results files of feature subsets data
│   └── Evaluation                                  <- Examples of processed results files of model evaluation
│
├── src                                             <- Source code for use in this project.
│   ├── d00_utils                                   <- Methods used across the project
│   │   └── calculateTimeDifference.py              <- Method to give time difference in minutes and seconds (string)
│   ├── d01_data                                    <- Scripts to reading and writing data
│   │   └── loadCommits.py                          <- Method to load .txt to a pandas dataframe
│   ├── d02_intermediate                            <- Scripts to transform data from raw to intermediate
│   │   ├── cleanCommitData.py                      <- Methods to clean SVN commit data
│   │   └── cleanJiraData.py                        <- Methods to clean JIRA data
│   ├── d03_processing                              <- Scripts to turn intermediate data into modelling input
│   │   ├── calculateCosineSimilarity.py            <- Methods to calculate cosine similarity between 2 documents
│   │   ├── calculateDocumentStatistics.py          <- Methods to compute document statistics features
│   │   ├── calculateQueryQuality.py                <- Methods to compute query quality features
│   │   ├── calculateTimeDif.py                     <- Method to calculate the time difference between 2 dates in seconds
│   │   ├── checkFullnameEqualsEmail.py             <- Method to check if the full_name is part of the email address
│   │   ├── checkValidityTrace.py                   <- Method to check if a trace is valid or invalid
│   │   ├── createCorpusFromDocumentList.py         <- Method to create a corpus
│   │   ├── createFittedTF_IDF.py                   <- Method to fit a tf_idf on a document
│   │   └── normalize_data.py                       <- Method to normalize data
│   └── d04_model_evaluation                        <- Scripts that analyse model performance and model selection
│   │   └── model_evaluation.py                     <- Methods to produce the evaluation metrics
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
5. Navigate to ```notebooks``` and execetute ```evaluation_notebook.ipynb```.
6. In the second cell block it is possible to set a number of evaluation rounds (default=2)
7. Navigate to ```Cell``` and then ```Run all cells```

For non-MDD specific features, the following steps shall be taken:
1. Navigate to ```notebooks``` and execetute ```lcd_trace_no_lc_features.ipynb```.
2. Navigate to ```Cell``` and then ```Run all cells```.

For automated feature selection, the following steps shall be taken:
1. Navigate to ```notebooks``` and execetute ```feature_selection.ipynb```.
2. In the second cell block it is possible to set a number of evaluation rounds (default=2) and a project name that is run.

# Publication
Van Oosten, W., Rasiman, R.S., Dalpiaz, F., & Hurkmans, T. (2022). On the Effectiveness of Automated Tracing Model Changes to Project Issues. [under review]
