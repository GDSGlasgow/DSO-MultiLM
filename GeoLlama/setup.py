#!/usr/bin/env python
from distutils.core import setup
from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt')

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

setup(name='GeoLlama',
      version='0.1',
      description='GeoLlama multi-lingual geoparser',
      author='Joe Shingleton',
      author_email='joseph.shingleton@glasgow.ac.uk',
      url='https://github.com/GDSGlasgow/DSO-MultiLM/tree/text-geoparsing-JS/GeoLlama',
      packages=reqs,
     )