import os
import pandas as pd
import json
import collections

class Metadata:
    """docstring for Metadata"""
    def __init__(self, location):
        self.location = location
        self.metadata_files = []
        self.metadata = {}
        self.metadata_dict = []
        self.create_metadata()

    def update(self, d, u):
        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                r = self.update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        return d

    def merge_dicts(self, dict_args):
        """
        Given an array of dicts, shallow copy and merge into a new dict,
        precedence goes to key value pairs in latter dicts.
        """
        result = {}
        for dictionary in dict_args:
            self.update(result, dictionary)
        return result

    def merge_parent(self, champ, champ_parent):
        """
        Fusionne le champ en cours et son parent de façon récursive
        """

        parent = self.metadata['champs'][champ_parent]
        this = self.metadata['champs'][champ]
        if not 'parent' in parent:
            return self.merge_dicts([parent, this])
        else:
            return self.merge_dicts([self.merge_parent(champ_parent, parent['parent'] ), this])

    def create_metadata(self):
        # Deal with metadata
        if self.location is not None:
            self.metadata_files.append(self.location)

        # Load as dict
        for f in self.metadata_files:
            with open(f, encoding="utf-8") as data_file:
                self.metadata_dict.append(json.load(data_file, encoding='utf-8'))

        # Merge metadata
        self.metadata = self.merge_dicts(self.metadata_dict)
        # Create parents if table

        # Create defaults behaviour
        default = self.metadata.pop('_default', {})

        # Update des champs parents
        for champ in self.metadata['champs']:
            if 'parent' in self.metadata['champs'][champ]:
                champ_parent = self.metadata['champs'][champ]['parent']
                self.metadata['champs'][champ] = self.merge_dicts([default, self.merge_parent(champ, champ_parent)])
            else:
                self.metadata['champs'][champ] = self.merge_dicts([default, self.metadata['champs'][champ]])

    def __call__(self):
        return pd.DataFrame(self.metadata.get('champs', {})).T.reset_index()

    def query(self, query="", exclude=[],include=[]):

        q = self().query(query)
        q = q.query('index not in @exclude')

        return list(q['index'].values)+include
