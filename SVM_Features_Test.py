from FeatureHolder import FeatureHolder
import sys
import pylab as pl
from sklearn import svm
import numpy as np

def main(argv):
#######################
### Basic Setup Things for pylab plot and feature holder
    a = FeatureHolder(argv[1])
    pl.figure(1)
    pl.clf()
    X_indices = np.arange(a.features.shape[-1])
######################
### Testing different kernels for weighting of features
    linear_svm1 = svm.SVC(kernel='linear',C=1)
    linear_svm1.fit(a.features,a.ids)
    linear_svm1_weights = (linear_svm1.coef_ ** 2).sum(axis=0)
    pl.bar(X_indices - .25, linear_svm1_weights, width=.2, label='Linear SVM with C=1 Weights', color = 'g')

    linear_svm2 = svm.SVC(kernel='linear',C=.1)
    linear_svm2.fit(a.features,a.ids)
    linear_svm2_weights = (linear_svm2.coef_ ** 2).sum(axis=0)
    pl.bar(X_indices - .45, linear_svm2_weights, width=.2, label='Linear SVM with C=.1 Weights', color = 'r')

    linear_svm3 = svm.SVC(kernel='linear',C=10)
    linear_svm3.fit(a.features,a.ids)
    linear_svm3_weights = (linear_svm3.coef_ ** 2).sum(axis=0)
    pl.bar(X_indices - .05, linear_svm3_weights, width=.2, label='Linear SVM with C=10 Weights', color = 'b')

    linear_svm4 = svm.SVC(kernel='linear',C=.01)
    linear_svm4.fit(a.features,a.ids)
    linear_svm4_weights = (linear_svm4.coef_ ** 2).sum(axis=0)
    pl.bar(X_indices - .65, linear_svm4_weights, width=.2, label='Linear SVM with C=.01 Weights', color = 'y')

    pl.title("Comparing feature weights with different SVM Models")
    pl.xlabel('Feature number')
    pl.yticks(())
    pl.axis('tight')
    pl.legend(loc='upper right')

    pl.show()





if __name__=='__main__':
    main(sys.argv)
