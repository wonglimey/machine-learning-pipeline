import preprocessing_functions as pf
import config

# =========== scoring pipeline =========

# impute categorical variables
def predict(data):
    
    # extract first letter from cabin
    data[config.EXTRACT_VARIABLE] = pf.extract_cabin_letter(data, config.EXTRACT_VARIABLE)


    # impute NA categorical
    for var in config.CATEGORICAL_TO_ENCODE:
        data[var] = pf.impute_na(data, var, replacement='Missing')
    
    
    # impute NA numerical
    for var in config.NUMERICAL_TO_IMPUTE:
        if (var == 'age'):
            data[var] = pf.add_missing_indicator(data, var, config.AGE_MEDIAN)
        else:
            data[var] = pf.add_missing_indicator(data, var, config.FARE_MEDIAN)
    
    
    # Group rare labels
    for var in config.CATEGORICAL_TO_ENCODE :
        data[var] = pf.remove_rare_labels(data, var, config.RARE_VALUE)
    
    
    # encode variables
    for var in config.CATEGORICAL_TO_ENCODE:
        data = pf.encode_categorical(data, var)
        
        
    # check all dummies were added
    pf.check_dummy_variables(data, config.DUMMY_VARIABLE)

    
    # scale variables
    data = pf.scale_features(data[config.FEATURES],
                             config.OUTPUT_SCALER_PATH)
    
    
    # make predictions
    predictions = pf.predict(data, config.OUTPUT_MODEL_PATH)
    
    return predictions


# ======================================
    
# small test that scripts are working ok
    
if __name__ == '__main__':
        
    from sklearn.metrics import accuracy_score    
    import warnings
    warnings.simplefilter(action='ignore')
    
    # Load data
    data = pf.load_data(config.PATH_TO_DATASET)
    
    X_train, X_test, y_train, y_test = pf.divide_train_test(data,
                                                            config.TARGET)
    
    pred = predict(X_test)
    
    # evaluate
    # if your code reprodues the notebook, your output should be:
    # test accuracy: 0.6832
    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
    print()
        