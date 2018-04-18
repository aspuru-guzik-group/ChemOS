#!/usr/bin/env python 

import glob
import pickle 
import numpy as np 

#==========================================

target = np.array([8., 35., 17])
target = target / np.sum(target)

#==========================================

data = {'params': [], 'values': []}

folders = ['color_runs']
for folder in folders:
	for file_index, file_name in enumerate(glob.glob('%s/*pkl' % folder)):
		
#		if file_index == 100:
#			break

		file_data = pickle.load(open(file_name, 'rb'))

		rgb = file_data['measured_colors_rgb']
		if np.linalg.norm(rgb) < 5.:
			continue
		rgb = rgb / np.sum(rgb)
		distance = np.linalg.norm(rgb - target)

		rgb = file_data['parameters'][:5]
#		rgb = rgb / np.sum(rgb)
#		rgb = rgb[:4]

		data['params'].append(rgb)
		data['values'].append(file_data['peak_area'])

data['params'] = np.array(data['params'])
data['values'] = np.array(data['values'])

pickle.dump(data, open('colors_green.pkl', 'wb'))
