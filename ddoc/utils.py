
from .reporting.word_generator import WordGenerator
from .reporting.excel_generator import ExcelGenerator
from .reporting.metadata import create_template
from .addons.addons import *
from .reporting.addon import *
import os
import subprocess

# Generate a "classical dict"
"""
This module is composed of high level methods for generating a report, it highly simplifies the call to WordGenerator
This is the main entry point of the package

Info: warning, une copie du df est place en parametre pour eviter de ressortir les variables discretes contruites.. attention au gros dataset
"""

def generate_excel(df, out_directory='', metadata_location=None, addons='none', **kwargs):
    """ Generic method to make a Excel Report

    Parameters
    ----------
        df: dataset, pd.DataFrame
            The pandas DataFrame to describe, or a location to csv file if it ends with .csv or pickle file otherwise
        out_directory: path, str
            the directory location where you want to output the report, default: current directory
        metadata_location: path, str
            The location of metadata. If None, automatically infer metadata and store it in the same folder that out location, default: None
    
    Example
    -------
        .. code-block:: python
    
            from ddoc import generate
            generate('your_file.csv')
            
    """
    if metadata_location is None:
        metadata_location = os.path.join(out_directory, 'metadata.json')
        create_template(df, metadata_location, **kwargs)

    generator = ExcelGenerator(table=df, out_directory=out_directory, metadata_location=metadata_location)
    generator.generate(**kwargs)

def generate(df, out_directory='', metadata_location=None, addons='none', **kwargs):
    """ Generic method to make a Word Report

    Parameters
    ----------
    
        df: data, pd.DataFrame
            The pandas DataFrame to describe, or a location to csv file if it ends with .csv or pickle file otherwise
        out_directory: path, str
            the directory location where you want to output the report, default: current directory
        metadata_location: path, str
            The location of metadata. If None, automatically infer metadata and store it in the same folder that out location, default: None
        addons: param, str
            Addons you want to put in the report, default:
            - 'all' : It will use a standard set of addons such as projection on target and automatic discretisation of numerical variables
            - 'none' : No addons
            - 'ratio': Only Addon with TARGET proportion
    
    Example
    -------
        .. code-block:: python
    
            from ddoc import generate
            generate('your_file.csv')
            
    """
    if metadata_location is None:
        metadata_location = os.path.join(out_directory, 'metadata.json')
        create_template(df, metadata_location, **kwargs)
    field_addons =[]
    if addons=='all':
        field_addons = [AddonIsTarget(), AddonIsTargetQuantile()] 
    elif addons=='all++': # V Cramer 
        field_addons = [AddonIsTarget(), AddonIsTargetQuantile()] 
        addons = [CorrelationAddon()]
    elif addons=='none':
        field_addons = [] 
    elif addons=='ratio':
        field_addons = [AddonIsTarget()] 
    elif addons=='quantile':
        field_addons = [AddonIsTargetQuantile()] 
    generator = WordGenerator(table=df, out_directory=out_directory, metadata_location=metadata_location, field_addons=field_addons, addons=addons)
    generator.generate(**kwargs)



