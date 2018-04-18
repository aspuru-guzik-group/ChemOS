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
_DB_RESULTS_PATH  = '%s/Experiments/Databases/results.db' % _HOME
_DB_RESULTS_TYPE  = 'sqlite'
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
					 'results_database': {'path': _DB_RESULTS_PATH,
					 					  'database_type': _DB_RESULTS_TYPE},
					 'feedback_database': {'path': _DB_FEEDBACK_PATH,
					 					   'database_type': _DB_FEEDBACK_TYPE},

#					 'communicator': {'type': 'auto'},
					 'communicator': {'type': 'slack'},


					 'experiments': [{'name': 'maximize HPLC peak area', 
					 				  'variables':  [{'name': 'param0', 'type': 'float', 'low': 0.0, 'high': 1.0, 'size': 1},
					 				  			     {'name': 'param1', 'type': 'float', 'low': 0.0, 'high': 1.0, 'size': 1},
												 	 {'name': 'param2', 'type': 'float', 'low': 0.0, 'high': 1.0, 'size': 1},
												 	 {'name': 'param3', 'type': 'float', 'low': 0.0, 'high': 1.0, 'size': 1},
												 	 {'name': 'param4', 'type': 'float', 'low': 0.0, 'high': 1.0, 'size': 1},
												 	 {'name': 'param5', 'type': 'float', 'low': 0.0, 'high': 1.0, 'size': 1}],

									  'objectives': [{'name': 'peak_area',      'operation': 'std_rel', 'hierarchy': 0, 'type': 'minimum', 'tolerance': 0.2},
									  				 {'name': 'peak_area',      'operation': 'average', 'hierarchy': 1, 'type': 'maximum', 'tolerance': 0.5},
									  				 {'name': 'sample_volume',  'operation': 'average', 'hierarchy': 2, 'type': 'minimum', 'tolerance': 0.2},
									  				 {'name': 'execution_time', 'operation': 'average', 'hierarchy': 3, 'type': 'minimum', 'tolerance': 0.5}],

					 				  'repetitions': 3,
									  'description': 'This experiment aims to mix available colors to create a solution of a defined target color'}],

					 'bots': [{'name': 'default_bot', 
					 		   'parameters': ['param0', 'param1', 'param2', 'param3', 'param4', 'param5'],
							   'communication': {'dump_path': '/home/chemos/Dropbox/chemos/input', 'pick_up_path': '/home/chemos/Dropbox/chemos/output',
							   					 'status_file': '/home/chemos/Dropbox/chemos/ChemOS_status.pkl',
#							   'communication': {'dump_path': '/home/chemos/Dropbox/chemos/input', 'pick_up_path': '/home/chemos/Dropbox/chemos/output',
#							   					 'status_file': '/home/chemos/Dropbox/chemos/ChemOS_status.pkl',
#					 		   'communication': {'dump_path': '/home/chemos/Dropbox/chemos/input', 'pick_up_path': '/home/chemos/Dropbox/chemos/input', 
#					 		   					 'username': 'john_doe', 'host': '192.168.0.1'}}]
												 }}]			 

					 												   }


