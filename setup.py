from setuptools import setup, find_packages
from pip.req import parse_requirements

reqs = [
        'pandas>=0.25.3', 
        'numpy>=1.18.0',
        'seaborn>=0.9.0',
        'matplotlib>=3.1.2', 
        'xlsxwriter>=1.2.7', 
        'scikit-learn>=0.22', 
        'python-docx>=0.8.10'
        ]

setup(name='ddoc',
      version='0.0.1', 
      description='Create documentation of your data in word or/and excel format', 
      url='https://github.com/data-corentinV/ddoc',
      author='see AUTHORS.rst file', 
      packages=find_packages(), 
      keywords=['data', 'documentation', 'word', 'excel', 'analysis'], 
      install_requires=reqs
      )