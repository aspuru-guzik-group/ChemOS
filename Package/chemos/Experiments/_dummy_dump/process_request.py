#!/usr/bin/env python 

import sys
import pickle
import numpy as np 

np.set_printoptions(precision = 4)

file_name = sys.argv[1]

data = pickle.load(open(file_name, 'rb'))


parameters = data['parameters']
print(parameters, np.linalg.norm(parameters))
data['loss'] = np.linalg.norm(parameters)

pickle.dump(data, open('../_dummy_pick_up/%s' % file_name, 'wb'))

