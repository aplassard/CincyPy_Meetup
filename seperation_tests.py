import sys
import numpy as np
import pylab as pl
from sklearn.svm import SVC
from FeatureHolder import FeatureHolder
from matplotlib.colors import ListedColormap

n_neighbors = 10
h = .02
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

def main(argv):
    a = FeatureHolder(argv[1])
    features = [5,42,8,18,16]
    Y = a.ids
    print
    for i in xrange(len(features)):
        for j in xrange(i+1,len(features)):
            X = a.features[:,[features[i],features[j]]]
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            pl.figure(1)
            clf = SVC(kernel='linear')
            clf.fit(X,Y)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            pl.pcolormesh(xx, yy, Z, cmap=cmap_light)
            pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
            t ="Linear Kernel SVM Test Comparing Features %s and %s with a score of %s" % (a.feature_names[features[i]],a.feature_names[features[j]], clf.score(X,Y))
            print t
            pl.title(t)
            pl.axis("tight")

            pl.figure(2)
            clf = SVC(kernel='rbf')
            clf.fit(X,Y)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            pl.pcolormesh(xx, yy, Z, cmap=cmap_light)
            pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
            t ="Gaussian Kernel SVM Test Comparing Features %s and %s with a score of %s" % (a.feature_names[features[i]],a.feature_names[features[j]], clf.score(X,Y))
            print t
            pl.title(t)
            pl.axis("tight")

            pl.figure(3)
            clf = SVC(kernel='sigmoid')
            clf.fit(X,Y)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            pl.pcolormesh(xx, yy, Z, cmap=cmap_light)
            pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
            t = "Sigmoid Kernel SVM Test Comparing Features %s and %s with a score of %s" % (a.feature_names[features[i]],a.feature_names[features[j]], clf.score(X,Y))
            print t
            pl.title(t)
            pl.axis("tight")

            pl.figure(4)
            clf = SVC(kernel='poly')
            clf.fit(X,Y)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            pl.pcolormesh(xx, yy, Z, cmap=cmap_light)
            pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
            t = "Polynomial Kernel SVM Test Comparing Features %s and %s with a score of %s" % (a.feature_names[features[i]],a.feature_names[features[j]], clf.score(X,Y))
            print t
            pl.title(t)
            pl.axis("tight")

            pl.show()
            print

    X = a.features[:,features]
    clf = SVC(kernel='linear')
    clf.fit(X,Y)
    print "Linear SVM Classifier using all %s Features had a score of %s" % (len(features),clf.score(X,Y),)
    clf = SVC(kernel='rbf')
    clf.fit(X,Y)
    print "Gaussian SVM Classifier using all %s Features had a score of %s" % (len(features),clf.score(X,Y),)
    clf = SVC(kernel='sigmoid')
    clf.fit(X,Y)
    print "Sigmoid SVM Classifier using all %s Features had a score of %s" % (len(features),clf.score(X,Y),)
    clf = SVC(kernel='poly')
    clf.fit(X,Y)
    print "Polynomial SVM Classifier using all %s Features had a score of %s" % (len(features),clf.score(X,Y),)
    print

    X = a.features
    clf = SVC(kernel='linear')
    clf.fit(X,Y)
    print "Linear SVM Classifier using all Features had a score of %s" % (clf.score(X,Y),)
    clf = SVC(kernel='rbf')
    clf.fit(X,Y)
    print "Gaussian SVM Classifier using all Features had a score of %s" % (clf.score(X,Y),)
    clf = SVC(kernel='sigmoid')
    clf.fit(X,Y)
    print "Sigmoid SVM Classifier using all Features had a score of %s" % (clf.score(X,Y),)
    clf = SVC(kernel='poly')
    clf.fit(X,Y)
    print "Polynomial SVM Classifier using all Features had a score of %s" % (clf.score(X,Y),)
    print

if __name__=='__main__':
    main(sys.argv)
