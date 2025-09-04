import openml
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from urllib.error import URLError


# def monkey_patch():
#     openml.datasets.functions._get_dataset_parquet = lambda x: None

def do_nothing(*args, **kwargs):
    return None

openml.datasets.functions._get_dataset_parquet = do_nothing


openml_cc18_list=[
    1485,50,1497,4538,1501,
    1489,40975,14,307,18,
    40984,54,11,182,6332,
    46,6,44,3,16,1487,188,
    151,40979,15,40701,38,
    1468,458,469,29,40670,
    1462,40966,28,1478,1475,
    31,32,40978,41027,37,
    1068,40983,1067,1590,1461,
    40923,4134,23517,1494,40994,
    1486,300,1063,40499,40982,
    4534,1053,1050,12,23,
    40668,1480,1510,1049,
    1464,22,23381]

our_list = [ 
    4538, 40975, 40701,  1497,   469,  
    1464,    23,  1475,  1067, 40982,  
    1068,  1050,  1049,  1487,  1494,  
    1063,  1510,   458, 1485,  4134]

our_new_list = our_list+ [40994, 1480, 11, 37, 54, 50, 31, 
                          # 1462 # very easy; give f1 of 1.00 on all methods
                          46
                         ] # easy 8

hard_list = [
    23,   151,  1049,  1050,  
    1067,  1068,  1461,  1464,  1475,
    1485,  1487,  1497,  4538, 40668, 
    40701, 40975, 40982, 41027]

large_dataset = [
    # 151, 1461, 40668, 41027
]

hard_list = np.setdiff1d(hard_list, large_dataset).tolist()


def get_data(id=31):
    '''return OHEncoded df'''
    dataset = openml.datasets.get_dataset(id, download_data=False)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X, columns=attribute_names)
    cat_mask = np.array(categorical_indicator)

    numeric_features = df.columns[~cat_mask]
    categorical_features = df.columns[cat_mask]

    # numeric_transformer = Pipeline(
    #     steps=[
    #         ("scaler", StandardScaler())
    #     ]
    # )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            # ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder='passthrough'
    )

    X_ohe = preprocessor.fit_transform(df)

    return X_ohe, y


def get_data1(id=31, tries=5):
    '''return df with categories are label encoded'''
    dataset = 0
    err = 0
    
    while(tries>0):
        try:
            dataset = openml.datasets.get_dataset(id, download_data=False)
            break
        except URLError as e:
            tries =- 1
            err = e
            
    if tries<=0:
        raise err

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X, columns=attribute_names)
    cat_mask = np.array(categorical_indicator)

    numeric_features = df.columns[~cat_mask]
    categorical_features = df.columns[cat_mask]

    df_copy = df.copy(deep=0)

    for i in categorical_features:
        le = LabelEncoder()
        df_copy[i] = le.fit_transform(df[i].values)
        # df_copy[i] = df_copy[i].astype('int32')

    return df_copy, y, cat_mask



def get_data_raw(id=31, tries=5):
    '''return df with categories are label encoded'''
    dataset = 0
    err = 0
    
    while(tries>0):
        try:
            dataset = openml.datasets.get_dataset(id, download_data=False)
            break
        except URLError as e:
            tries =- 1
            err = e
            
    if tries<=0:
        raise err

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X, columns=attribute_names)
    cat_mask = np.array(categorical_indicator)

    numeric_features = df.columns[~cat_mask]
    categorical_features = df.columns[cat_mask]

    df_copy = df.copy(deep=0)

    return df_copy, y, cat_mask


def get_data_from_featllm(dataset, cv=True, full_shot = True):
    file_name = f"../featllm/data/{dataset}.csv"
    
    df = pd.read_csv(file_name)
    default_target_attribute = df.columns[-1]
    
    categorical_indicator = [True if (dt == np.dtype('O') or pd.api.types.is_string_dtype(dt)) else False for dt in df.dtypes.tolist()][:-1]
    attribute_names = df.columns[:-1].tolist()

    X = df.convert_dtypes()
    y = df[default_target_attribute].to_numpy()
    X = df.drop(columns = default_target_attribute)
    label_list = np.unique(y).tolist()
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return X, y, np.array(categorical_indicator)