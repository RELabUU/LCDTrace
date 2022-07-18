# Feature Selection 2.0
This notebook is created to perform automated feature selection using the MRMR feature selection algorithm

## Installation
For this notebook, the package of MRMR needs to be installed which can be done through the following command in the terminal:

```bash
pip install mrmr_selection
```


## Use of the notebook
The notebook is currently set up to run a single dataset (non-normalised) and put it three times through the MRMR algorithm, being for K=40, K=50 and K=60. For each one, it performs the feature selection for that specifc K-value, followed by the evaluation.py script and finally it processes the results form that to get the averages of the evaluation run. 

## Parameters to configure
Before running the notebook, three parameter can/should be adjusted, being:
1. dataset : string // which is used to properly write the output files
2. n_runs : integer // number of runs of the evalution script (default = 10, not necessary to change)
3. top_n : integer // The top-n of feature importances (default = 10, not necessary to change)

## Output 
The notebook delivers several files being, for each K-value that is being processed:
1. The subset of the non-normalised dataset that is created by the MRMR algorihm (filename = {dataset}_subset_K{number}.xlsx
2. A textfile listing only the selected features (filename = {dataset}_K{number}_selected_features.txt
3. An excel file containing the averages of the selected feature importance scores for both the XGBoost and LightGBM algorithm (filename = {dataset}_importance_subset_K{number}.xlsx

Besides the files listed above for each K-value, the notebook delivers for all processed K-values a merged file, being:
1. An excel file with the F-scores (averaged evaluation runs with n=10) for all K-values
2. An excel file with the top-10 based on the feature importances

# Upcoming
Notebook must receive an update for the (possible) next step, which is running more specific K-values after the first analysis on the K=40, K=50 and K=60. This could be simply adjusting the K-values as they are currently encoded
