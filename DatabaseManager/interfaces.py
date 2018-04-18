#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import json
import sqlalchemy as sql 

from sqlalchemy.types import TypeDecorator
from sqlalchemy.ext.mutable import Mutable

#========================================================================

class JSONdict(TypeDecorator):

	impl = sql.Text(256)

	def process_bind_param(self, value, dialect):
		if value is not None:
			value = json.dumps(value)
		return value

	def process_result_value(self, value, dialect):
		if value is not None:
			value = json.loads(value)
		return value

#========================================================================

class MutableDict(Mutable, dict):
	
	@classmethod
	def coerce(cls, key, value):
		if not isinstance(value, MutableDict):
			if isinstance(value, dict):
				return MutableDict(value)
			return Mutable.coerce(key, value)
		else:
			return value

	def __setitem__(self, key, value):
		dict.__setitem__(self, key, value)
		self.changed()

	def __delitem__(self, key):
		dict.__delitem__(self, key)
		self.changed()


#========================================================================

class SQLiteDatabase(object):

	SQLITE_COLUMNS = {'string': sql.String(256), 
					  'pickle': sql.PickleType(),
					  'integer': sql.Integer(),
					  'float': sql.Float(),
					  'dictionary': MutableDict.as_mutable(JSONdict)}


	def __init__(self, path, attributes, name = 'table', verbose = True):

		self.db_path    = 'sqlite:///%s' % path
		self.attributes = attributes
		self.name       = name
		self.verbose    = verbose

		# create database
		self.db = sql.create_engine(self.db_path)
		self.db.echo = False
		self.metadata = sql.MetaData(self.db)

		# create table 
		self.table = sql.Table(self.name, self.metadata)
		for name, attribute_type in self.attributes.items():
			self.table.append_column(sql.Column(name, self.SQLITE_COLUMNS[attribute_type]))
		self.table.create(checkfirst = True)




	def _add(self, new_entries):
		with self.db.connect() as connection:
			connection.execute(self.table.insert(), new_entries)
			connection.close()



	def _fetch(self, condition_dict):

		condition_keys   = list(condition_dict.keys())
		condition_values = list(condition_dict.values())

		# defining the selection
		selection = sql.select([self.table]).where(getattr(self.table.c, condition_keys[0]) == condition_values[0])
		for index, key in enumerate(condition_keys[1:]):
			selection = selection.where(getattr(self.table.c, key) == condition_values[index + 1])

		# getting one entry meeting the selection criteria
		with self.db.connect() as connection:
			selected = connection.execute(selection)
			entry    = selected.fetchone()
			connection.close()

		return entry




	def _fetch_all(self, condition_dict):

		condition_keys   = list(condition_dict.keys())
		condition_values = list(condition_dict.values())

		# defining the selection
		selection = sql.select([self.table]).where(getattr(self.table.c, condition_keys[0]) == condition_values[0])
		for index, key in enumerate(condition_keys[1:]):
			selection = selection.where(getattr(self.table.c, key) == condition_values[index + 1])

		# get all entries
		with self.db.connect() as connection:
			selected = connection.execute(selection)
			entries  = selected.fetchall()
			connection.close()
		return entries



	def _purge(self, condition_dict):

		condition_keys   = list(condition_dict.keys())
		condition_values = list(condition_dict.values())
	
		purge = sql.delete(self.table).where(getattr(self.table.c, condition_keys[0]) == condition_values[0])
		for index, key in enumerate(condition_keys[1:]):
			purge = purge.where(getattr(self.table.c, key) == condition_values[index + 1])

		with self.db.connect() as connection:
			purged = connection.execute(purge)
			connection.close()
		return 0



	def _update(self, condition_dict, update_dict):

		condition_keys   = list(condition_dict.keys())
		condition_values = list(condition_dict.values())

		# defining the selection 
		update = sql.update(self.table).values(update_dict).where(getattr(self.table.c, condition_keys[0]) == condition_values[0])
		for index, key in enumerate(condition_keys[1:]):
			update = update.where(getattr(self.table.c, key) == condition_values[index + 1])
		
		# get all entries
		with self.db.connect() as connection:
			updated = connection.execute(update)
			connection.close()
		return 0


