
from matplotlib import pyplot
from numpy import where
from collections import Counter

def summariseClassDistribution(features, labels):
    # summarize class distribution
    counter = Counter(labels)
    print(counter)
    
    # scatter plot of examples by class label
    for label, _ in counter.items():
        row_ix = where(labels == label)[0]
        pyplot.scatter(features[row_ix, 0], features[row_ix, 1], label=str(label))
    pyplot.legend()
    pyplot.show()