import pandas as pd
from typing import List
from sklearn import base
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from hamilton.function_decorators import extract_columns


def _sanitize_columns(
    df_columns: List[str]  # the current column names
) -> List[str]:  # sanitized column names
    return [c.strip().replace("/", "_per_").replace(" ", "_").lower() for c in df_columns]


@extract_columns(*['passengerid',
                   'survived',
                   'pclass',
                   'name',
                   'sex',
                   'age',
                   'sibsp',
                   'parch',
                   'ticket',
                   'fare',
                   'cabin',
                   'embarked'])
def input_data(index_col: str, location: str) -> pd.DataFrame:
    """
    
    Here are the features in the data:
        survived - Survival (0 = No; 1 = Yes)
        class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
        name - Name
        sex - Sex
        age - Age
        sibsp - Number of Siblings/Spouses Aboard
        parch - Number of Parents/Children Aboard
        ticket - Ticket Number
        fare - Passenger Fare
        cabin - Cabin
        embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
        boat - Lifeboat (if survived)
        body - Body number (if did not survive and body was recovered)
    
    :param index_col:
    :param location: 
    :return:
    """
    df = pd.read_csv(location)
    df.columns = _sanitize_columns(df.columns)
    df = df.set_index(index_col)
    df = df.fillna(0)  # fill NA here.
    return df


def cabin_t(
    cabin: pd.Series # raw cabin info
) -> pd.Series: # transformed cabin info
    return cabin.apply(lambda x: x[:1] if x is not np.nan else np.nan)


def ticket_t(
    ticket: pd.Series # raw ticket number
) -> pd.Series: # transformed ticket number
    return ticket.apply(lambda x: str(x).split()[0])


def family(
    sibsp: pd.Series, # number of siblings
    parch: pd.Series # number of parents/children
) -> pd.Series: # number of people in family
    return sibsp + parch


def _label_encoder(
    input_series: pd.Series # series to categorize
) -> preprocessing.LabelEncoder: # sklearn label encoder
    le = preprocessing.LabelEncoder()
    le.fit(input_series)
    return le

def _label_transformer(
    fit_le: preprocessing.LabelEncoder, # a fit label encoder
    input_series: pd.Series # series to transform 
) -> pd.Series: # transformed series
    return fit_le.transform(input_series)


def sex_encoder(sex: pd.Series) -> preprocessing.LabelEncoder:
    return _label_encoder(sex)

def cabin_encoder(cabin: pd.Series) -> preprocessing.LabelEncoder:
    return _label_encoder(cabin)

def embarked_encoder(embarked: pd.Series) -> preprocessing.LabelEncoder:
    return _label_encoder(embarked)

def sex_category(sex: pd.Series, sex_encoder: preprocessing.LabelEncoder) -> pd.Series:
    return _label_transformer(sex_encoder, sex)

def cabin_category(cabin: pd.Series, cabin_encoder: preprocessing.LabelEncoder) -> pd.Series:
    return _label_transformer(cabin_encoder, cabin)

def embarked_category(embarked: pd.Series, embarked_encoder: preprocessing.LabelEncoder) -> pd.Series:
    return _label_transformer(embarked_encoder, embarked)

