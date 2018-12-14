#!/usr/bin/env python 

import os
import time
import shutil
import glob
import pickle 
import numpy as np 

#===================================

def run_experiments():

	while True:

		file_names = glob.glob('*rep*pkl')

		for file_name in file_names:

			exp_dict = pickle.load(open(file_name, 'rb'))

			params = exp_dict['parameters']
			objs   = {}
		
			exp_dict['obj_0'] = np.random.uniform(low = 0, high = 1)
			exp_dict['obj_1'] = np.random.uniform(low = 0, high = 1)
			exp_dict['obj_2'] = np.random.uniform(low = 0, high = 1)
			
#			exp_dict['obj_0'] = np.sum(np.square(params))
#			exp_dict['obj_1'] = np.sum(np.square(params))
#			exp_dict['obj_2'] = np.sum(np.square(params))
	
			pickle.dump(exp_dict, open('../output/%s' % file_name, 'wb'))	
			print(exp_dict)
			print('========')
			os.remove(file_name)

			break

		time.sleep(2)

#===================================




if __name__ == '__main__':

	run_experiments()


