#!/usr/bin/env python 

import glob
import pickle 
import numpy as np 

#==========================================

data = {'params': [], 'values': []}

folders = ['random_search', 'spearmint_run']
for folder in folders:
	for file_index, file_name in enumerate(glob.glob('%s/*pkl' % folder)):
		
#		if file_index == 100:
#			break

		file_data = pickle.load(open(file_name, 'rb'))

		data['params'].append(file_data['parameters'])
		data['values'].append(file_data['peak_area'])

data['params'] = np.array(data['params'])
data['values'] = np.array(data['values'])

pickle.dump(data, open('n9_data.pkl', 'wb'))
