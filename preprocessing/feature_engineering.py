import pandas as pd
import numpy as np
#minmaxscaler scales numeric features 
#it takes every value in a column and relates it to range between 0-1
#Like annual_inc = range from 10k-500k -> 10k->0.0 , 500k->1.0 and everytthing else lands in between
from sklearn.preprocessing import MinMaxScaler
# splits the data into a training set and test set
# train the model on training set and eval on test set
# The test set can create new test cases (loans) the model has never seen ->measures if model is actually learning or not
from sklearn.model_selection import train_test_split
import os
import pickle
# saves Python objects to disk as binary files.
#  You'll use it to save the scaler so that when a new loan application comes in later,
#  you can scale it the exact same way you scaled the training data.

#Data_DIR and OUT_DIR point to data/processed since we are reading parquet files we created and writing 
# the scaled versions back to same folder
DATA_DIR  = 'data/processed'
OUT_DIR   = 'data/processed'

#ALL features accounted for except Home ownership-> cant scle it with a range
NUMERIC_FEATURES = [
    'loan_amnt', 'int_rate', 'installment', 'annual_inc',
    'dti', 'fico_range_low', 'fico_range_high', 'inq_last_6mths',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
    'total_acc', 'delinq_2yrs', 'emp_length',
]
CATEGORICAL_FEATURES = ['home_ownership']
TARGET = 'loan_status' #The 0 snd 1 we are predicting


#Loops through banks a, b,c and bank.upper is for "Bank A instead of Bank b" type of statement
# pd.read_parquet statement loads each banks parquet file bank by bank and build filename dynamically 
for bank in ['a', 'b', 'c']:
    print(f"\nProcessing Bank {bank.upper()}...")

    df = pd.read_parquet(f'{DATA_DIR}/bank_{bank}.parquet')

    #x is everything ecvept the target column 
    X = df.drop(columns=[TARGET])
    y = df[TARGET]# pulls that one column

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
    # test_size=0.2 means 20% of the data goes to the test set, 80% goes to training.
    # random_state=42 makes the split reproducible. Without this, every time you run the script you'd get a different random split. 
    # Setting a fixed seed means always get the exact same split, important for consistency when you're comparing results across runs.
    # stratify is important because without it a random split might put one of the defaults intpo the training data and leave test set with no defaults
    # returns X_train, X_test, y_train and y_test, this is training data for model to learn from

    # create a new scaler for this bank — each bank gets its own because their data ranges differ
    scaler = MinMaxScaler()

    # fit learns the min/max of each column from training data, transform scales it to 0-1
    # only fit on training data — never on test data
    X_train[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])

    # scale test data using the SAME min/max learned from training — no fitting here
    # this prevents data leakage — in production you won't know future data's min/max
    X_test[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])

    # save scaled training features for this bank
    X_train.to_parquet(f'{OUT_DIR}/bank_{bank}_X_train.parquet', index=False)

    # save scaled test features for this bank
    X_test.to_parquet(f'{OUT_DIR}/bank_{bank}_X_test.parquet', index=False)

    # y_train is a Series not a DataFrame so convert it first before saving
    y_train.to_frame().to_parquet(f'{OUT_DIR}/bank_{bank}_y_train.parquet', index=False)

    # same for y_test
    y_test.to_frame().to_parquet(f'{OUT_DIR}/bank_{bank}_y_test.parquet', index=False)

    # save the scaler object so we can scale new loan applications the same way later
    # wb = write binary — pickle files are binary not text
    with open(f'{OUT_DIR}/bank_{bank}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # confirm row counts and default rates look correct for each bank
    print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"  Default rate train: {1 - y_train.mean():.1%} | test: {1 - y_test.mean():.1%}")