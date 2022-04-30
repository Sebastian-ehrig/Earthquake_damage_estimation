import pandas as pd
from typing import List
from tqdm import tqdm
import category_encoders as ce

def encoder(values: pd.DataFrame, labels: pd.DataFrame, string_columns: List[str], one_hot_columns: List[str]) -> pd.DataFrame:
    """
    This function reads in the values and the labels, applies target encoding and returns
    a dataframe which is consists only of numeric columns.
    """

    train_data = pd.merge(values, labels, how="inner")
    encoded_values = values.copy()  

    # target encoding with label-mean
    for col in tqdm(string_columns):
        data_snippet = train_data.loc[:, [col, "damage_grade"]].copy()
        categorical_dict = data_snippet.groupby(col).mean().to_dict()["damage_grade"]
        encoded_values.loc[:, col] = values.loc[:, col].map(categorical_dict).astype(float)


    # one-hot encoding
    # create an object of the OneHotEncoder
    ce_one = ce.OneHotEncoder(cols=one_hot_columns) 
    encoded_values = ce_one.fit_transform(encoded_values)


    #encoded_values = encoded_values.fillna(encoded_values.mean())
    encoded_values = encoded_values.fillna(encoded_values.mean())

    assert all(encoded_values.dtypes != str)
    assert encoded_values.isna().any().any() == False, "There are missing values in encoded data"

    return encoded_values
