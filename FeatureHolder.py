import numpy as np

class FeatureHolder(object):
    '''
        Simple Class to Hold Information About the Features
    '''

    def __init__(self,filename):
        f = open(filename,'r')
        features = []
        self.labels   = []
        self.ids      = []
        line = f.readline()
        line = line.strip().split(',')
        self.feature_names = line[2:]
        self._int_dict = {}
        n = 0
        for line in f:
            line = line.strip().split(',')
            key = self._int_dict.get(line[1],n)
            if key == n: n+=1
            self._int_dict[line[1]]=key
            self.labels.append(line[0])
            self.ids.append(key)
            features.append(line[2:])
        self.features = np.array(features,dtype=np.float)
