#!/usr/bin/env python 

import time
import numpy as np 
import subprocess

from file_logger import FileLogger


np.set_printoptions(precision = 4)


def process(file_name):
#	print('found new file: ', file_name)
	print('...')
	time.sleep(10)
	subprocess.call('./process_request.py %s' % file_name, shell = True)
	print('done')

logger = FileLogger(action = process, path = './')
while True:
	time.sleep(1)
