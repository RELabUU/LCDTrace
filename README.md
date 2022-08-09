# Online appendix 
**Keywords:** Requirements Traceability - Trace Link Recovery - Model-Driven Development - Low-Code Development - Machine Learning

## About
This file presents the online appendix which contains the files to produce the results of Van Oosten, Rasiman, Dalpiaz & Hurkmans (2022). The online appendix is defided in two parts, being (1) the tools required for producing the results (LCDTrace) and (2) the files for each research question (RQ) specifically. A readme.md file specified for the LCDTrace tool can be found in the directory itself. The LCDTrace folder is equivalant to the Github as referred to in the paper. 

# Directory Structure
The project follows the structure laid out by [Data Science for Social Good](https://github.com/dssg/hitchhikers-guide). Below is a modified of their system as used in this project
```
├── LCDTrace                                        <- Files necessary to produce the results for the research (notebooks and example data)
├── RQ1                                             <- Jupyter notebooks
│   ├── 1.Trace Link Feature Data                   <- Populated feature files for the datasets used in this research
│   │   2. Non-normalised Results                   <- Evaluation output for non-normalised features for the datasets used in this research
│   │   4. Feature Importance Results               <- Evaluation output for feature importances for the datasets used in this research 
│   │   RQ1_F-metrics.xlsx                          <- File that presents aggregated results from non-normalized results (Table 8)
│   │   RQ1_Feature_Importance_analysis.xlsx        <- File that is used to analyse the feature importance of feature-level
│   └── RQ1_Statistics.xlsx                         <- File that contains results of statistical test (Table 6 and 7)
├── RQ2                                             <- The evaluation results produced by the notebook
│   ├── 1.Trace Link Feature Data                   <- Populated feature files for the datasets used in this research
│   │   2. Non-normalised Results                   <- Evaluation output for non-normalised features for the datasets used in this research
│   │   4. Feature Importance Results               <- Evaluation output for feature importances for the datasets used in this research 
│   │   RQ2_F-metrics.xlsx                          <- File that presents aggregated results from non-normalized results (Table 10b)
│   │   RQ2_Feature_Importance_analysis.xlsx        <- File that is used to analyse the feature importance of feature-level
│   └── RQ2_Feature_Importance.xlsx                 <- File that presents aggregated results from feature importance results (Table 11)
├── RQ3                                             <- Source code for use in this project.
│   ├── K40
│   │   ├── 1.Trace Link Feature Data               <- Populated feature files for the datasets used in this research
│   │   │   2. Non-normalised Results               <- Evaluation output for non-normalised features for the datasets used in this research
│   │   │   4. Feature Importance Results           <- Evaluation output for feature importances for the datasets used in this
│   │   │   RQ3_Feature_Importance_analysis_K40.xlsx<- File that is used to analyse the feature importance of feature-level
│   ├── K50
│   │   ├── 1.Trace Link Feature Data               <- Populated feature files for the datasets used in this research
│   │   │   2. Non-normalised Results               <- Evaluation output for non-normalised features for the datasets used in this research
│   │   │   4. Feature Importance Results           <- Evaluation output for feature importances for the datasets used in this
│   │   │   RQ3_Feature_Importance_analysis_K50.xlsx<- File that is used to analyse the feature importance of feature-level 
│   ├── K60
│   │   ├── 1.Trace Link Feature Data               <- Populated feature files for the datasets used in this research
│   │   │   2. Non-normalised Results               <- Evaluation output for non-normalised features for the datasets used in this research
│   │   │   4. Feature Importance Results           <- Evaluation output for feature importances for the datasets used in this
│   │   │   RQ3_Feature_Importance_analysis_K60.xlsx<- File that is used to analyse the feature importance of feature-level
│   ├── RQ3_F-metrics.xlsx                          <- File presents the aggregated results of the evaluation script on the classifiers (Table 12)
│   ├── RQ3_Feature_Families_Importance.xlsx        <- File presents the aggregated feature importances per feature family (Table 14)
│   └── RQ3_Statistics.xlsx                         <- File that contains results of statistical test regarding Table 13
└── readme.md                 
```

# Publication
Van Oosten, W., Rasiman, R.S., Dalpiaz, F., & Hurkmans, T. (2022). On the Effectiveness of Automated Tracing Model Changes to Project Issues. [under review]
