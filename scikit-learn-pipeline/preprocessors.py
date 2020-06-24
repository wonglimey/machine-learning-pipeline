import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from sklearn.base import BaseEstimator, TransformerMixin


# Add binary variable to indicate missing values
class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accommodate sklearn pipeline functionality
        return self

    def transform(self, X):
        # add indicator
        X = X.copy()
        
        for var in self.variables:
            X[var+'_NA'] = np.where(X[var].isnull(), 1, 0)
        return X


# Categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        
        for var in self.variables:
            X[var].fillna('Missing', inplace=True)
        return X


# Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        
        for var in self.variables:
            self.imputer_dict_[var] = X[var].median()
        return self

    def transform(self, X):
        X = X.copy()
        
        for var in self.variables:
            X[var] = X[var].fillna(self.imputer_dict_[var])
        return X


# Extract first letter from string variable
class ExtractFirstLetter(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        
        for var in self.variables:
            X[var] = X[var].str[0]
        return X
        

# Frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):
        self.tol = 0.05
        
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}
        
        for var in self.variables:
            # the encoder will learn the most frequent categories
            var_count = pd.Series(X[var].value_counts() / np.float(len(X)))
            # create a list of frequent labels
            self.encoder_dict_[var] = list(var_count[var_count >= self.tol].index)
        return self

    def transform(self, X):
        X = X.copy()
        
        # replace rare variables with 'Rare' value
        for var in self.variables:
            X[var] = np.where(X[var].isin(self.encoder_dict_[var]), X[var], 'Rare')
        return X


# String to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # HINT: persist the dummy variables found in train set
        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        self.missing_dummies = []
        # get dummies
        for var in self.variables:
            X = pd.concat([X, pd.get_dummies(X[var], prefix=var, drop_first=True)], axis=1)
        # drop original variables
        X.drop(labels=self.variables, axis=1, inplace=True)
        # find the missing dummy variables
        for var in self.dummies:
            if var not in list(X.columns):
                self.missing_dummies.append(var)
         # add missing dummies if any
        if self.missing_dummies:
            for var in self.missing_dummies:
                X[var] = 0      
        return X
