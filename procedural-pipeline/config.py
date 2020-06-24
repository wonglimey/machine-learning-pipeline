# ====   PATHS ===================

PATH_TO_DATASET = "titanic.csv"
OUTPUT_SCALER_PATH = 'scaler.pkl'
OUTPUT_MODEL_PATH = 'logistic_regression.pkl'


# ======= PARAMETERS ===============
# imputation parameters
AGE_MEDIAN = 28
FARE_MEDIAN = 14.45

EXTRACT_VARIABLE = 'cabin'


# encoding parameters
RARE_VALUE = 0.05

DUMMY_VARIABLE = 'embarked_Rare'

FREQUENT_LABELS = {
    'sex': ['female', 'male'],
    'cabin': ['C', 'Missing'],
    'embarked': ['C', 'Q', 'S'],
    'title': ['Miss', 'Mr', 'Mrs']}


# ======= FEATURE GROUPS =============
TARGET = 'survived'

NUMERICAL_TO_IMPUTE = ['age', 'fare']

CATEGORICAL_TO_ENCODE = ['sex', 'cabin', 'embarked', 'title']


# selected features for training
FEATURES = ['pclass', 'age', 'sibsp', 'parch',
           'fare', 'age_NA', 'fare_NA', 'sex_male',
           'cabin_Missing', 'cabin_Rare', 'embarked_Q', 'embarked_Rare',
           'embarked_S', 'title_Mr', 'title_Mrs', 'title_Rare']



