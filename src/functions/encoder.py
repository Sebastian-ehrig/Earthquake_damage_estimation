import pandas as pd
from typing import List
from tqdm import tqdm

def encoder(values: pd.DataFrame, labels: pd.DataFrame, string_columns: List[str]) -> pd.DataFrame:
    """
    This function reads in the values and the labels, applieds target encoding and returns
    a dataframe which is consists only of numeric columns
    """

    train_data = pd.merge(values, labels, how="inner")
    encoded_values = values.copy()

    for col in tqdm(string_columns):
        data_snippet = train_data.loc[:, [col, "damage_grade"]].copy()
        categorical_dict = data_snippet.groupby(col).mean().to_dict()["damage_grade"]
        encoded_values.loc[:, col] = values.loc[:, col].map(categorical_dict).astype(float)

    assert all(encoded_values.dtypes != str)
    return encoded_values