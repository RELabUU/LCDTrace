#Import Python Libraries
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve

from imblearn.pipeline import Pipeline 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

#Method to show the different model evaluation metrics
def showModelPerformance(trainedModel, testFeatures, testLabels):
    # Use the fitted model to predict the labels of the test set
    predictionLabels = trainedModel.predict(testFeatures)
    
    #Calculate the different metrics for the test vs predicted labels
    accuracyValue = accuracy_score(testLabels.astype(bool), predictionLabels)
    precisionValue = precision_score(testLabels.astype(bool), predictionLabels, average='binary')
    f1Value = f1_score(testLabels.astype(bool), predictionLabels)
    f2Value = fbeta_score(testLabels.astype(bool), predictionLabels, beta=2.0)
    f05Value = fbeta_score(testLabels.astype(bool), predictionLabels, beta=0.5)
    recallValue = recall_score(testLabels.astype(bool), predictionLabels)
    averagePrecisionValue = average_precision_score(testLabels.astype(bool), predictionLabels)
    
    #Create a dataframe to output all evaluation metrics in
    performanceData = {'Accuracy':  [accuracyValue],
                       'Precision': [precisionValue],
                       'Recall': [recallValue],
                       'F1': [f1Value],
                       'F2': [f2Value],
                       'F0.5': [f05Value],
                       'Average Precision': [averagePrecisionValue]
                      }
    performanceDf = pd.DataFrame(performanceData)
    return(performanceDf)

#Method to define the Pipeline steps based on the given rebalancing strategy and classification algorithm
def define_steps(rebalancing_strategy, classification_algorithm):
    steps = None
    if(rebalancing_strategy == 'none'):
        if(classification_algorithm == 'random_forests'):
            steps = [['classifier', RandomForestClassifier(n_jobs=-1)]]
        elif (classification_algorithm == 'xg_boost'):
            steps = [['classifier', xgb.XGBClassifier(n_jobs=-1)]]
            return(steps)
        elif(classification_algorithm == 'light_gbm'):
            steps = [['classifier', lgb.LGBMClassifier(n_jobs=-1, importance_type='gain')]]
    elif(rebalancing_strategy == 'over'):
        if(classification_algorithm == 'random_forests'):
            steps = [['smote', SMOTE()],
                    ['classifier', RandomForestClassifier(n_jobs=-1)]]
        elif (classification_algorithm == 'xg_boost'):
            steps = [['smote', SMOTE()],
                    ['classifier', xgb.XGBClassifier(n_jobs=-1)]]
        elif(classification_algorithm == 'light_gbm'):
            steps = [['smote', SMOTE()],
                    ['classifier', lgb.LGBMClassifier(n_jobs=-1, importance_type='gain')]]
    elif(rebalancing_strategy == 'under'):
        if(classification_algorithm == 'random_forests'):
            steps = [['under', RandomUnderSampler()],
                    ['classifier', RandomForestClassifier(n_jobs=-1)]]
        elif (classification_algorithm == 'xg_boost'):
            steps = [['under', RandomUnderSampler()],
                    ['classifier', xgb.XGBClassifier(n_jobs=-1)]]
        elif(classification_algorithm == 'light_gbm'):
            steps = [['under', RandomUnderSampler()],
                    ['classifier', lgb.LGBMClassifier(n_jobs=-1, importance_type='gain')]]
    elif(rebalancing_strategy == '5050'):
        if(classification_algorithm == 'random_forests'):
            steps = [['smote', SMOTE(sampling_strategy = 0.5)],
                    ['under', RandomUnderSampler()],
                    ['classifier', RandomForestClassifier(n_jobs=-1)]]
        elif (classification_algorithm == 'xg_boost'):
            steps = [['smote', SMOTE(sampling_strategy = 0.5)],
                    ['under', RandomUnderSampler()],
                    ['classifier', xgb.XGBClassifier(n_jobs=-1)]]
        elif(classification_algorithm == 'light_gbm'):
            steps = [['smote', SMOTE(sampling_strategy = 0.5)],
                    ['under', RandomUnderSampler()],
                    ['classifier', lgb.LGBMClassifier(n_jobs=-1, importance_type='gain')]]
    return steps

#Method to generate the f1, f2, f0.5, accuracy, precision, recall, and average precision
def generate_evaluation_metrics(rebalancing_strategy, classification_algorithm, data, labels, is_normalized, n_runs, feature_names, n_features):
    #Create a dataframe to append to the results of each individual run
    evaluation_df = pd.DataFrame(
    {
        'Accuracy':  [],
        'Precision': [],
        'Recall': [],
        'F1': [],
        'F2': [],
        'F0.5': [],
        'Average Precision': []
    })
    
    #Create a np array to put the importances per feature in
    importance_array = np.empty(shape=(n_runs, n_features))
    
    #Perform the described pipeline steps to produce the results for the defined number of runs
    for i in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(data,
                                                        labels,
                                                        test_size=0.2,
                                                        stratify=labels)
        
        #Set the pipeline steps according to the defined rebalancing strategy and classification algorithm
        steps = define_steps(rebalancing_strategy, classification_algorithm)
        
        #Create the pipeline
        model_pipeline = Pipeline(steps=steps)
        
        space_empty = dict()    
        
        stratified_kfold = StratifiedKFold(n_splits=10,shuffle=True)           
    
        #Create a model
        model = RandomizedSearchCV(estimator = model_pipeline, 
                                param_distributions = space_empty, 
                                n_iter=1, 
                                n_jobs=-1, 
                                cv = stratified_kfold)
        
        #Fit the model on the training data
        fitted_model = model.fit(X_train, y_train)
        
        #Evaluate the fitted model
        fitted_model_evaluation_df = showModelPerformance(trainedModel = fitted_model, 
                         testFeatures = X_test, 
                         testLabels = y_test)     
        
        #Add the evaluation of the current run to the results of the previous runs
        evaluation_df = pd.concat([evaluation_df,
                                   fitted_model_evaluation_df])
        
        #Find the feature importances of the fitted model
        if(classification_algorithm == "light_gbm"):
            current_importances = fitted_model.best_estimator_._final_estimator.booster_.feature_importance(importance_type='gain')
        else:
            current_importances = fitted_model.best_estimator_._final_estimator.feature_importances_
        #Add the feature importances of the current fitted model to the results of the previous runs
        importance_array[i] = current_importances  
    
    if is_normalized == True:
        dir_string = "3. Normalised Results"
    else:
        dir_string = "2. Non-Normalised Results"
    
    #Set the index as the run number
    evaluation_df = evaluation_df.reset_index(drop = True)
    evaluation_df.index += 1 
    evaluation_df.index.name = "run"
    
    #Output the evaluation data to a csv file
    evaluation_df.to_csv("../results/" + dir_string + "/" + classification_algorithm + "/" + rebalancing_strategy + "_results.csv")
    
    #Transform the importance array to a data frame
    importance_df = pd.DataFrame(data=importance_array, 
                                 columns= feature_names, 
                                 index=list(range(1, n_runs +1)))
    
    #Set the index as the run number
    importance_df.index.name = "run"
    
    #Output the importance data to a csv file
    importance_df.to_csv("../results/4. Feature Importance Results/" + classification_algorithm + "/" + rebalancing_strategy + "_results.csv")