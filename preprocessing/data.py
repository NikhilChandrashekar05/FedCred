import pandas as pd
import os



#the location of the raw CSV on your machine
locationcsv = 'data/accepted_2007_to_2018Q4.csv'

#where bank files will be saved post split
OUT_DIR = 'data/processed'

#16 features total
FEATURES = ['loan_amnt', 'int_rate' , 'installment' , 'annual_inc' , 'dti' , 
'fico_range_low', 'fico_range_high', 'inq_last_6mths',
'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
'total_acc', 'delinq_2yrs', 'emp_length', 'home_ownership',]

#target is what we are predicting
TARGET = 'loan_status'

#grade which used for splitting, never will get into the mdoel
SPLIT_KEY = 'grade'


print("Loading the data: ")
df = pd.read_csv(locationcsv, low_memory = False, usecols = FEATURES + [TARGET, SPLIT_KEY])
#usecols is useful because of dataset having 151 columns, but Im using about 16 features
#and the features +... tells pandas to ignore every other column while reading file

df = df[df[TARGET].isin(['Fully Paid', 'Charged Off'])].copy()
df[TARGET] = (df[TARGET] == 'Fully Paid').astype(int)
#converts numbers into results 

df['emp_length'] = df['emp_length'].str.extract(r'(\d+)').astype(float)
#whole bunch of conversion from messy text to clean numeric clm

df['home_ownership'] = df['home_ownership'].map(
    {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 3, 'NONE': 3}
)#Everywhere there is RENT then it addresses it with 0 etc.
# Process of Label Encoding- Conversion of categories to numbers in such for this tree based model

df = df.dropna(subset=FEATURES)

bank_a = df[df[SPLIT_KEY].isin(['A', 'B'])].drop(columns=SPLIT_KEY)
bank_b = df[df[SPLIT_KEY].isin(['C', 'D'])].drop(columns=SPLIT_KEY)
bank_c = df[df[SPLIT_KEY].isin(['E', 'F', 'G'])].drop(columns=SPLIT_KEY)

#. This is where dataset becomes three separate banks.
# grade is essentially a summary of all the other features —
#  FICO score, income, DTI. If you left it in, the model would just learn
#  "grade A = good, grade G = bad" and ignore everything else
#After these 3 lines of code 3 seperate dataframes are maintained, 
#Bank A sees data, Bank B sees its data only etc. 
#Which is the whole idea of Federated Learning


#creates the data/processed folder it it doesnt exist
os.makedirs(OUT_DIR, exist_ok= True)

#to.parquet is the saving process as a parquet file designed for tabular data
#bank a = grades A/B(lowest default rate)
bank_a.to_parquet(f'{OUT_DIR}/bank_a.parquet', index = False)
#bank b = grades C/D(Medium default rate)
bank_b.to_parquet(f'{OUT_DIR}/bank_b.parquet', index=False)
#bank c = grades E/F/G(Highest defauly rate)
bank_c.to_parquet(f'{OUT_DIR}/bank_c.parquet', index=False)

print(f"Bank A: {len(bank_a):,} rows | default rate: {1 - bank_a['loan_status'].mean():.1%}")
print(f"Bank B: {len(bank_b):,} rows | default rate: {1 - bank_b['loan_status'].mean():.1%}")
print(f"Bank C: {len(bank_c):,} rows | default rate: {1 - bank_c['loan_status'].mean():.1%}")

print(f"\nSaved to {OUT_DIR}/")