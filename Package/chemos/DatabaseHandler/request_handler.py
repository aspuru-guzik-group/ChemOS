#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import uuid

from DatabaseManager.database import Database
from Utils.utils import Printer

#========================================================================

class RequestHandler(Printer):

	DB_ATTRIBUTES = {'status': 'string',
					 'job_id': 'string', 
					 'exp_identifier': 'string',
					 'parameters': 'pickle',
					 'author': 'pickle'}


	def __init__(self, settings, verbose = True):
		Printer.__init__(self, 'REQUEST HANDLER', color = 'yellow')
		self.settings = settings
		self.verbose  = verbose
		self._create_database()


	def _create_database(self):
		db_settings   = self.settings['request_database']
		self.database = Database(db_settings['path'], self.DB_ATTRIBUTES,
								 db_settings['database_type'], verbose = self.verbose)


	def process_request(self, request_dict):
		# add information to dictionary
		job_id = str(uuid.uuid4())
		request_dict['job_id'] = job_id
		request_dict['status'] = 'pending'

		# dump entry in database
		self.database.add(request_dict)


	def remove_requests(self, identifier):
		self._print('removing requests for %s' % identifier)
		condition = {'exp_identifier': identifier}
		self.database.remove_all(condition)


	def _relabel(self, job_id, status):
		condition = {'job_id': job_id}
		update    = {'status': status}
		self.database.update(condition, update)
		return 0


	def label_processing(self, request_dict):
		return self._relabel(request_dict['job_id'], 'processing')


	def label_feedbackless(self, request_dict):
		return self._relabel(request_dict['job_id'], 'feedbackless')


	def label_done(self, request_dict):
		return self._relabel(request_dict['job_id'], 'done')


	def dump_parameters(self, request, parameters):
		self._print('dumping parameter set')
		print('\t\t', parameters)
		condition = {'job_id': request['job_id']}
		update    = {'parameters': parameters}
		self.database.update(condition, update)


	def get_pending_request(self):
		condition = {'status': 'pending'}
		requests  = self.database.fetch_all(condition)
		return requests

#========================================================================




#	def get_feedbackless_request(self, author):
#		condition = self.request_table.c.author_contact == author['contact']
#		selection = sql.select([self.request_table]).where(condition)
#		with self.db.connect() as request_conn:
#			selected = request_conn.execute(selection)
#			selected_new_request = selected.fetchone()
#			request_conn.close()
#		request_info_dict = self._entry_to_dict(selected_new_request)
#		return request_info_dict


#	def get_request_author(self, job_id):
#		condition = self.request_table.c.job_id == job_id
#		selection = sql.select([self.request_table]).where(condition)
#		with self.db.connect() as request_conn:
#			selected = request_conn.execute(selection)
#			selected_request = selected.fetchone()
#			request_conn.close()
#		info_dict = self._entry_to_dict(selected_request)
#		return info_dict


#==================================================================

if __name__ == '__main__':
	# IMPLEMENT YOUR TEST CASES HERE!
	fake = RequestHandler('sqlite://///home/flo/BobTheBot/DrinkOS/Databases/requests.db')
	fake.process_drink_request('tequilasunrise', {'contact': 'hase.florian@gmail.com'})

