# LCDTrace
**Keywords:** Requirements Traceability - Trace Link Recovery - Model-Driven Development - Low-Code Development - Machine Learning

## About
This repository provides the code used to produce the results of Rasiman, Dalpiaz & España (2022). It takes a set of JIRA issues, and SVN commits as input. The program then cleans and preprocesses these. The JIRA id (label) is then taken from the SVN commit logs and appended to the commit data using REGEX. The Cartesian product of the JIRA issues and SVN commits is then produced, with each element being a candidate trace. For each candidate trace, a set of features is computed. These features are then utilized as input data for 12 different models (classification algorithm x rebalancing strategy). Finally, these models are evaluated.

# Directory Structure
The project follows the structure laid out by [Data Science for Social Good](https://github.com/dssg/hitchhikers-guide). Below is a modified of their system as used in this project
```
├── LCDTrace                                        <- Files necessary to produce the results for the research (notebook and example data)
├── RQ1                                             <- Jupyter notebooks
│   ├── 1.Trace Link Feature Data                   <- Populated feature files for the datasets used in this research
│   │   2. Non-normalised Results                   <- Evaluation output for non-normalised features for the datasets used in this research
│   │   4. Feature Importance Results               <- Evaluation output for feature importances for the datasets used in this research 
│   │   RQ1_F-metrics.xlsx                          <- File that presents aggregated results from non-normalized results
│   └── RQ1_Feature_Importance.xlsx                 <- File that presents aggregated results from feature importance results
├── RQ2                                         <- The evaluation results produced by the notebook
│   ├── 1.Trace Link Feature Data                   <- Populated feature files for the datasets used in this research
│   │   2. Non-normalised Results                   <- Evaluation output for non-normalised features for the datasets used in this research
│   │   4. Feature Importance Results               <- Evaluation output for feature importances for the datasets used in this research 
│   │   RQ2_F-metrics.xlsx                          <- File that presents aggregated results from non-normalized results
│   └── RQ2_Feature_Importance.xlsx                 <- File that presents aggregated results from feature importance results
├── RQ3                                             <- Source code for use in this project.
│   ├── K40
│   │   ├── 1.Trace Link Feature Data               <- Populated feature files for the datasets used in this research
│   │   │   2. Non-normalised Results               <- Evaluation output for non-normalised features for the datasets used in this research
│   │   │    4. Feature Importance Results          <- Evaluation output for feature importances for the datasets used in this
│   ├── K50
│   │   ├── 1.Trace Link Feature Data               <- Populated feature files for the datasets used in this research
│   │   │   2. Non-normalised Results               <- Evaluation output for non-normalised features for the datasets used in this research
│   │   │    4. Feature Importance Results          <- Evaluation output for feature importances for the datasets used in this 
│   ├── K60
│   │   ├── 1.Trace Link Feature Data               <- Populated feature files for the datasets used in this research
│   │   │   2. Non-normalised Results               <- Evaluation output for non-normalised features for the datasets used in this research
│   │   │    4. Feature Importance Results          <- Evaluation output for feature importances for the datasets used in this 
└── readme.md                 
```

# Publication
Rasiman, R.S., Dalpiaz, F., & España, S. (2022). How Effective Is Automated Trace Link Recovery in Model-Driven Development? In *International Working Conference on Requirements Engineering: Foundation for Software Quality*. Springer, Nature.
