#!/usr/bin/env python 

__author__ = 'Florian Hase'


#========================================================================

from DatabaseManager.interfaces import SQLiteDatabase

#========================================================================

class Database(object):

	def __init__(self, path, attributes, database_type = 'sqlite', name = 'table', verbose = True):

		self.db = None
		self.attributes = attributes

		if database_type == 'sqlite':
			self.db = SQLiteDatabase(path, attributes, name, verbose)
		else:
			raise NotImplementedError()



	def _entry_to_dict(self, entry):
		info_dict = {key: entry[key] for key in self.attributes}
		return info_dict



	def add(self, dictionary):
		self.db._add(dictionary)



	def fetch(self, condition, target = None):

		# fetches the entire entry
		entry = self.db._fetch(condition)

		if not target:
			return entry

		# gets the value corresponding to the target
		try:
			value = entry[target]
		except TypeError:
			value = None
		return value




	def fetch_all(self, condition, target = None):

		raw_entries = self.db._fetch_all(condition)
		entries = [self._entry_to_dict(entry) for entry in raw_entries]
		if not target:
			return entries

		values = []
		for entry in entries:
			try: 
				values.append(entry[target])
			except KeyError:
				pass
		return values



	def remove_all(self, condition):
		self.db._purge(condition)



	def replace(self, condition, new_entries):
		# first, purge all old entries
		self.db._purge(condition)

		# then add all entries
		self.db._add(new_entries)



	def update(self, condition, update):
		return self.db._update(condition, update)


		
