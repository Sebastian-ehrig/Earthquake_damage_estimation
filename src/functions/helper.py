
import pandas as pd
from pandas_profiling import ProfileReport

def load_data(path):
    df = pd.read_csv(path)
    return df

def change_datatype(data, string_columns, int_columns):
    data.loc[:, string_columns] = data.loc[:, string_columns].astype(str)
    data.loc[:, int_columns] = data.loc[:, int_columns].astype(int)
    return data

def data_profiler(data, name):
    """
    This function makes a pandas profilier report on 'data'. 
    The report is stored as html in reports. 
    """
    prof = ProfileReport(data)
    prof.to_file(output_file=f'./reports/data_report_{name}.html')
    pass