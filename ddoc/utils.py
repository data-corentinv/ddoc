
from .reporting.word_generator import WordGenerator
from .reporting.excel_generator import ExcelGenerator
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
    """
    Generic method to make a Excel Report
    :param df: The pandas DataFrame to describe, or a location to csv file if it ends with .csv or pickle file otherwise
    :param out_directory: the directory location where you want to output the report, default: current directory
    :param metadata_location: The location of metadata. If None, automatically infer metadata and store it in the same folder that out location, default: None
    """
    if metadata_location is None:
        metadata_location = os.path.join(out_directory, 'metadata.json')
        create_template(df, metadata_location, **kwargs)

    generator = ExcelGenerator(table=df, out_directory=out_directory, metadata_location=metadata_location)
    generator.generate(**kwargs)

def generate(df, out_directory='', metadata_location=None, addons='none', **kwargs):
    """
    Generic method to make a Word Report
    :param df: The pandas DataFrame to describe, or a location to csv file if it ends with .csv or pickle file otherwise
    :param out_directory: the directory location where you want to output the report, default: current directory
    :param metadata_location: The location of metadata. If None, automatically infer metadata and store it in the same folder that out location, default: None
    :param addons: Addons you want to put in the report, default:
        - 'all' : It will use a standard set of addons such as projection on target and automatic discretisation of numerical variables
        - 'none' : No addons
        - 'ratio': Only Addon with TARGET proportion
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


def create_template(df, out_location, **kwargs):
    import json
    """
    Infer a metadata template for dataset
    """
    if isinstance(df, str):
        if(df.endswith('.csv')):
            df = pd.read_csv(df, **kwargs)
        else:
            df = pd.read_pickle(df)
    result = {"description": "Attention les types ont été inférrés automatiquement, ils sont peut-être faux"}
    champs = {}
    for col in df.columns:
        champs[col] = {
            'type': "numerique" if df[col].dtype.kind in 'biufc' else "categoriel",
            "sous-type": "float" if df[col].dtype.kind in 'biufc' else "string"
        }

    result['champs'] = champs
    res = json.dumps(result, indent=4)
    with open(out_location, 'w') as f:
        f.write(res)
