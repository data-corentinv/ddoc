

import os
import collections
import json
from datetime import datetime
import numpy as np
import pandas as pd
from .metadata import Metadata


#from .addon import AddonManager
cwd = os.getcwd()

type_list = {
            int: "Entier",
            float: "Nombre décimal",
            str: "String",
            type(datetime.now()): "Date",
            np.int32: "Entier",
            np.int64: "Entier",
            np.float32: "Nombre décimal",
            np.float64: "Nombre décimal"
}

class ReportGenerator:

    def get_metadata(self):
        """
        Retourne les metadata
        """
        return self.metadata

    def get_type(self, df, column):
        """
        Get type of a column in string format
        """
        type_in_df = type(df[column].iloc[0])
        if type_in_df in type_list:
            return type_list[type_in_df]
        else:
            return str(type_in_df)

    def get_type_from_sous_type(self, sous_type):
        """
        Retourne les types python en fonction du type inscrit dans les metadata
        """
        if sous_type=="string":
            return str
        if sous_type=="float":
            return np.float64
        if sous_type=="integer":
            return np.float64
        if sous_type=="date":
            return str
        else:
            return str

    def load_df(self, **kwargs):
        """
        Load the df using the metadata

        :param table: nom de la table utilisée
        """

        # Load file
        if isinstance(self.table, pd.DataFrame):
            df = self.table.copy()
        elif self.table.lower().endswith('.csv') or self.table.lower().endswith('.txt'):
            metadata = self.metadata
            dtypes = {}

            if metadata is not None:
                fields = metadata.get('champs', {})
                for key in fields:
                    if fields[key].get('type', 'unknown')=="categoriel":
                        dtypes[key] = "category"
                    else:
                        dtypes[key] = self.get_type_from_sous_type(fields[key].get('sous-type', 'string'))
            df = pd.read_csv(self.table, dtype=dtypes, **kwargs)

        else:
            df = pd.read_pickle(self.table)

        return df

    def get_usable_columns(self, df):
        """
        Filtre permettant de ne garder que les colonnes utilisables provenant du dictionnaire de données
        """
        metadata = self.metadata
        metadata = metadata.get('champs', {})
        columns = [column for column in df.columns if metadata.get(column, {}).get('alerte', '')=='utilisable']
        return df[columns]

    def __init__(self, out_directory='', table=None, metadata_location=None, field_addons=[], addons=[]):
        """
        :param out_directory: directory the ReportGenerator will output
        :param metadata_location: location of JSON Metadata
        :param table: Dataframe or path to pickled dataframe or csv
        """
        self.table = table # Dataframe or path to pickled dataframe or csv
        self.out_directory = out_directory
        #self.addons_manager =  AddonManager(addons, field_addons)
        self.field_addons = field_addons
        self.addons = addons
        self.metadata_raw = Metadata(metadata_location)
        self.metadata = self.metadata_raw.metadata

    def generate(self, **kwargs):
        raise NotImplementedError
