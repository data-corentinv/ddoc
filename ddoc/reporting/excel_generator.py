import sys
sys.path.insert(1, 'XlsxWriter-master')
import xlsxwriter
import pandas as pd
import numpy as np
import json
import ntpath
import os
from pdb import set_trace as pause

from .report_generator import ReportGenerator 



# On peut utiliser le fichier JSON pour remplir : La provenance
def get_percentage_missing(series):
    num = series.isnull().sum()
    den = series.shape[0]
    return 100*(num/den)

class ExcelGenerator(ReportGenerator):
    def __init__(self, table=None, out_directory='', metadata_location=None):
        super().__init__(table=table, out_directory=out_directory, metadata_location=metadata_location)
        self.workbook = xlsxwriter.Workbook(os.path.join(self.out_directory, 'macro_report.xlsx'))
        self.columns = ['champ', 'commentaire', 'type', 'origine', 'provenance', 'sous-type','concentration', 'taux de NA', 'motif-rejet' ]
        #self.columns = ['Indicateurs', 'Description', 'Origine', 'Provenance', 'Nombre de valeurs distinctes', 'Valeurs manquantes']
        
        self.max_value_categorical = 500

    def get_style(self, *classes):
        res = {}
        for c in classes:
            res.update(self.styles.get(c, {}))
        return self.workbook.add_format(res)

    def configurate(self):
        # Define styles

        self.styles = {

        'header': {'bold': 1, 'bg_color': '#A589D9', 'font_size': 14},
        'field' : {'font_size': 12},
        'high-warning' : {'bold': 1, 'bg_color': '#F16D64'},
        'medium-warning' : {'bold': 1, 'bg_color': '#F3C746'},
        'low-warning' : {'bold': 1, 'bg_color': '#FBE491'},
        'parfait' : {'bold': 1,  'bg_color': '#95C753'},
        'ok' : {'bold': 1, 'bg_color': '#E0F4BE'},
        'réfléchir' : {'bold': 1, 'bg_color': '#F59640'},
        'modelisable' :{'bold': 1, 'bg_color': '#BFABE6'},
        'utilisable' : {'bold': 1, 'bg_color': '#95C753'},
        'inutilisable' : {'bold': 1, 'bg_color': '#F16D64'},
        'categoriel' : {'bold': 1, 'bg_color': '#CFEDFB'},
        'continue'  : {'bold': 1, 'bg_color': '#EBE4FF'},
        'id' : {'bold': 1, 'bg_color': '#FFE0DA'},
        'inconnu' : {'bold': 1, 'bg_color': '#FFE7BB'},
        'numerique' :  {'bold': 1, 'bg_color': '#D2ECEB'},
        'float' : {'bold': 1, 'bg_color': '#FFF2B6'},
        'integer' : {'bold': 1, 'bg_color': '#FFDFF2'},
        'string' : {'bold': 1, 'bg_color': '#E0F4BE'},
        'date' : {'bold': 1, 'bg_color': '#E6E9EC'},
        'full_border' : {'border' : 1, 'top': 1, 'bottom':1 , 'left': 1, 'right' : 1},
        'centered' : {'align' : 'center', 'valign' : 'vcenter'},
        'text' : {'italic' : 1 }
        }

    def generate(self, **kwargs):

        # Create a workbook and add a worksheet.
        """
        For each data file in directory create a type description
        """
        self.configurate()
        
        try:
            sheet_name_original = ntpath.basename(self.table)
        except:
            sheet_name_original = "myDataset"
        sheet_name = sheet_name_original[:min(31, len(sheet_name_original))]
        worksheet = self.workbook.add_worksheet(sheet_name)
        # Load file
        df = self.load_df(**kwargs)
        self.get_types_report(df, worksheet, sheet_name_original)
        self.workbook.close()

    def get_types_report(self, df, worksheet, table, **kwargs):
        metadata = self.metadata
        self.define_headers(df, worksheet, dict(metadata.get('champs', {})))
        self.fill_fields(df, worksheet, dict(metadata.get('champs', {})))
        
    def fill_fields(self, df, worksheet, metadata):
        for idx_field, field in enumerate(sorted(df.columns)):
            for idx_column, column in enumerate(self.columns):
                if column=='type':
                    worksheet.write(idx_field+1, idx_column, metadata.get(field, {}).get('type', self.get_type(df, field)),self.get_style('centered', 'full_border', metadata.get(field, {}).get(column, 'inconnu')) )
                elif column=='taux de NA':
                    percentage_missing = get_percentage_missing(df[field])
                    if percentage_missing==0:
                        style = self.get_style('centered', 'full_border', 'parfait')
                    elif percentage_missing<1:
                        style = self.get_style('centered', 'full_border', 'ok')
                    elif percentage_missing<10:
                        style = self.get_style('centered', 'full_border', 'low-warning')
                    elif percentage_missing<50:
                        style = self.get_style('centered', 'full_border', 'medium-warning')
                    else:
                        style = self.get_style('centered', 'full_border', 'high-warning')
                    worksheet.write(idx_field+1, idx_column, str(percentage_missing), style)
                elif column=='provenance':
                    worksheet.write(idx_field+1, idx_column, metadata.get(field, {}).get('provenance', ""))
                elif column=='sous-type':
                    worksheet.write(idx_field+1, idx_column, metadata.get(field, {}).get('sous-type', self.get_type(df, field)), self.get_style('centered', 'full_border', metadata.get(field, {}).get(column, 'inconnu')))
                elif column=='concentration':
                    worksheet.write(idx_field+1, idx_column, len(df[field].unique()), self.get_style('centered') )
                elif column=='commentaire':
                    worksheet.write(idx_field+1, idx_column, metadata.get(field, {}).get('description', ''), self.get_style('text') )
                elif column=='origine':
                    worksheet.write(idx_field+1, idx_column, metadata.get(field, {}).get('origine', ''), self.get_style('text') )
                elif column=='provenance':
                    worksheet.write(idx_field+1, idx_column, metadata.get(field, {}).get('provenance', ''), self.get_style('text') )
                elif column=='motif-rejet':
                    if metadata.get(field, {}).get('alerte')=='inutilisable':
                        worksheet.write(idx_field+1, idx_column, metadata.get(field, {}).get('warning', ''), self.get_style('text') )

    def define_headers(self, df, worksheet, metadata):
        """
        Define name of rows and column
        """
        fields = sorted(df.columns)
        worksheet.set_row(0,25)
        for idx, field in enumerate(fields):
            worksheet.set_row(idx+1,25)
            worksheet.write(idx+1, 0, field, self.get_style('field', 'centered', 'full_border', metadata.get(field, {}).get('alerte', None)))
        for idx, column in enumerate(self.columns):
            worksheet.set_column(idx,0, 25)
            worksheet.write(0, idx, column, self.get_style('centered','full_border', 'header'))

        
