
import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df

def change_datatype(data, string_columns, int_columns):
    data.loc[:, string_columns] = data.loc[:, string_columns].astype(str)
    data.loc[:, int_columns] = data.loc[:, int_columns].astype(int)
    return data

        