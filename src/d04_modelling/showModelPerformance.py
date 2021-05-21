from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve

import pandas as pd

def showModelPerformance(trainedModel, testFeatures, testLabels):
    # Use the forest's predict method on the test data
    predictionLabels = trainedModel.predict(testFeatures)
    
    accuracyValue = accuracy_score(testLabels.astype(bool), predictionLabels)
    precisionValue = precision_score(testLabels.astype(bool), predictionLabels, average='binary')
    f1Value = f1_score(testLabels.astype(bool), predictionLabels)
    f2Value = fbeta_score(testLabels.astype(bool), predictionLabels, beta=2.0)
    f05Value = fbeta_score(testLabels.astype(bool), predictionLabels, beta=0.5)
    recallValue = recall_score(testLabels.astype(bool), predictionLabels)
    averagePrecisionValue = average_precision_score(testLabels.astype(bool), predictionLabels)
        
    print('Legend')
    print('Recall -  measures the fraction of relevant links that are retrieved')
    print('Precision - measures the fraction of retrieved links that are relevant')
    print('F-measure - measures the harmonic mean of recall and precision')
    print('F2-measure - favors recall') 
    print('F0.5-measure - favors precision')
    print('Average Precision - summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold')
    print('')
          
    performanceData = {'Accuracy':  [accuracyValue],
                       'Precision': [precisionValue],
                       'Recall': [recallValue],
                       'F1': [f1Value],
                       'F2': [f2Value],
                       'F0.5': [f05Value],
                       'Average Precision': [averagePrecisionValue]
                      }
    performanceDf = pd.DataFrame (performanceData)
    print(performanceDf)
    
    plot_confusion_matrix(trainedModel, testFeatures, testLabels, colorbar=False, labels=[True,False])
    print('')
    print('--------------------------------------------------------------------')
    plot_roc_curve(trainedModel, testFeatures, testLabels) 
    plot_precision_recall_curve(trainedModel, testFeatures, testLabels) 