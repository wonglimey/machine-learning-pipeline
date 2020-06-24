import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    return pd.read_csv(df_path)


def divide_train_test(df, target):
    # Function divides data set in train and test
    X_train, y_train, X_test, y_test = train_test_split(df, df[target], test_size=0.2, random_state=0)
    return X_train, y_train, X_test, y_test
    

def extract_cabin_letter(df, var):
    # captures the first letter
    return df[var].str[0]  


def add_missing_indicator(df, var, median):
    # function adds a binary missing value indicator
    df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)
    return df[var].fillna(median)

    
def impute_na(df, var, replacement='Missing'):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    return df[var].fillna(replacement)


def remove_rare_labels(df, var, rare_value):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    encoder_dict_ = {}
    var_count = pd.Series(df[var].value_counts() / np.float(len(df)))
    encoder_dict_[var] = list(var_count[var_count >= rare_value].index)
    return np.where(df[var].isin(encoder_dict_[var]), df[var], 'Rare')


def encode_categorical(df, var):
    # adds ohe variables and removes original categorical variable
    df = pd.concat([df, pd.get_dummies(df[var], prefix=var, drop_first=True)], axis=1)
    df.drop(labels=var, axis=1, inplace=True)
    return df


def check_dummy_variables(df, dummy_list):
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    if dummy_list not in df.columns:
        df[dummy_list] = 0
    

def train_scaler(df, output_path):
    # train and save scaler
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler
      

def scale_features(df, scaler):
    # load scaler and transform data
    scaler = joblib.load(scaler)
    return scaler.transform(df)


def train_model(df, target, output_path):
    # initialise the model
    reg_model = LogisticRegression(C=0.0005, random_state=0)
    
    # train the model
    reg_model.fit(df, target)
    
    # save the model
    joblib.dump(reg_model, output_path)
    
    return None


def predict(df, model):
    # load model and get predictions
    model = joblib.load(model)
    return model.predict(df)
