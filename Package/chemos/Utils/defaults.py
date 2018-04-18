#!/usr/bin/env python 

import os 

#==============================================================

_HOME = os.getcwd()

_DB_PARAM_PATH    = '%s/Experiments/Databases/parameters.db' % _HOME
_DB_PARAM_TYPE    = 'sqlite'
_DB_BOT_PATH      = '%s/Experiments/Databases/bots.db' % _HOME
_DB_BOT_TYPE      = 'sqlite'
_DB_REQUEST_PATH  = '%s/Experiments/Databases/requests.db' % _HOME
_DB_REQUEST_TYPE  = 'sqlite'
_DB_FEEDBACK_PATH = '%s/Experiments/Databases/feedback.db' % _HOME
_DB_FEEDBACK_TYPE = 'sqlite'

_ALGORITHM_NAME = 'phoenics'
#_ALGORITHM_NAME = 'random_search'
#_ALGORITHM_NAME = 'spearmint'
#_ALGORITHM_NAME = 'smac'

#==============================================================

_DEFAULT_SETTINGS = {'algorithm': {'name': _ALGORITHM_NAME,
								   'num_batches': 1,
								   'batch_size': 2,
								   'random_seed': 100691},	

					 'account_details': {},	

					 'scratch_dir': '%s/.scratch' % _HOME,

					 'bot_database': {'path': _DB_BOT_PATH,
					 				  'database_type': _DB_BOT_TYPE},
					 'param_database': {'path': _DB_PARAM_PATH, 
								   		'database_type': _DB_PARAM_TYPE},
					 'request_database': {'path': _DB_REQUEST_PATH,
					 					  'database_type': _DB_REQUEST_TYPE},
					 'feedback_database': {'path': _DB_FEEDBACK_PATH,
					 					   'database_type': _DB_FEEDBACK_TYPE},

					 'communicator': {'type': 'slack'},


					 'experiments': [{'name': 'colors', 'variables': [{'name': 'param0', 'type': 'float', 'low': 0.0, 'high': 1.0, 'size': 1},
					 												   {'name': 'param1', 'type': 'float', 'low': 0.0, 'high': 1.0, 'size': 1},
																	   {'name': 'param2', 'type': 'float', 'low': 0.0, 'high': 1.0, 'size': 1}],
					 				  'loss_name': 'distance', 'loss_type': 'minimum', 'repetitions': 1,
									  'description': 'This experiment aims to mix available colors to create a solution of a defined target color'}],

					 'bots': [{'name': 'default_bot', 
					 		   'parameters': ['param0', 'param1', 'param2', 'param3'],
							   'communication': {'dump_path': '/home/chemos/Dropbox/chemos/input', 'pick_up_path': '/home/chemos/Dropbox/chemos/output',
							   					 'status_file': '/home/chemos/Dropbox/chemos/ChemOS_status.pkl',
#					 		   'communication': {'dump_path': '/home/chemos/Dropbox/chemos/input', 'pick_up_path': '/home/chemos/Dropbox/chemos/input', 
#					 		   					 'username': 'john_doe', 'host': '192.168.0.1'}}]
												 }}]			 

					 												   }


