#!/usr/bin/env python

import os
from setuptools import setup, find_packages

home = os.path.abspath(os.path.dirname(__file__))

def readme():
	with open('README.rst') as f:
		return f.read()


setup(name             = 'chemos',
	  version          = '0.1a8',
	  description      = 'This is an informative description of ChemOS',      
	  long_description = readme(),
	  url              = 'https://github.com/aspuru-guzik-group/chemos',
	  author           = 'Florian Hase, Loic Roch', 
	  author_email     = 'fhase@g.harvard.edu',
	  license          = 'Apache',
	  packages         = find_packages(),
	  install_requires = [
	  		'Flask',
	  		'lazyme', 
	  		'numpy',
	  		'slackclient',
	  		'sqlalchemy',
	  		'watchdog'
	  ],
	  include_package_data = True,
	  zip_safe     = False,
	  )
