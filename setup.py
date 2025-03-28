#!/usr/bin/env python

import pathlib
import pkg_resources
import setuptools
import subprocess
from setuptools.command.install import install
import shutil

# Check if Git is installed
if not shutil.which("git"):
    raise EnvironmentError("Git is not installed or not found in PATH")

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

class CustomInstallCommand(install):
    def run(self):
        # Install the xformers package with custom index URL
        subprocess.check_call([
            'pip', 'install', 'xformers<0.0.26', '--index-url', 'https://download.pytorch.org/whl/cu121'
        ])
        # Install the other dependencies from requirements.txt
        for requirement in install_requires:
            subprocess.check_call(['pip', 'install', str(requirement)])
        super().run()

setuptools.setup(
    name='MultiLM',
    version='0.1',
    description='Multi-Modal Multi-Lingual Location Extraction',
    author='Joe Shingleton, Yunus Serhat Bicakci, Yu Wang',
    author_email='joseph.shingleton@glasgow.ac.uk',
    url='https://github.com/GDSGlasgow/DSO-MultiLM/',
    packages=setuptools.find_packages(where='package'),
    package_dir={'': 'DSO-MultiLM'},
    install_requires=[],
    cmdclass={
        'install': CustomInstallCommand,
    },
)