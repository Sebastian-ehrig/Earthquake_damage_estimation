
import pandas as pd


import os


def load_data(path):
    df = pd.read_csv(path)
    return df