import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

with open("params.yaml" , "r") as f:
    params = yaml.safe_load(f)


test_size = params['data_ingestion']['test_size']
random_state = params['data_ingestion']['random_state']




# data/raw/messy_customer_churn_2000.csv"
def ingest_data(url):
    try:
        df = pd.read_csv(url)
    except FileNotFoundError:
        print("File Not Found Error....")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return df

def drop_duplicated(df):
    df.drop_duplicates(inplace=True)
    df.shape
    return df


# age

def age_cleaning(df):
    df['age'] = df['age'].astype(str).str.strip()
    df['age'] = df['age'].str.replace(' yrs', '', regex=False)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df.loc[~df['age'].between(18, 100), 'age'] = np.nan
    df['age'].isnull().sum()

    return df

def tenure_months(df):
    df['tenure_months'] = df['tenure_months'].astype(str).str.strip()
    df['tenure_months'] = df['tenure_months'].str.replace(' years', '', regex=False)
    df['tenure_months'] = df['tenure_months'].str.replace(' year', '', regex=False)
    df['tenure_months'] = pd.to_numeric(df['tenure_months'], errors='coerce')
    df.loc[~df['tenure_months'].between(1, 72), 'tenure_months'] = np.nan
    df['tenure_months'].isnull().sum()

    return df


def monthly_charges(df):
    df['monthly_charges'] = df['monthly_charges'].astype(str).str.strip()
    df['monthly_charges'] = df['monthly_charges'].str.replace('$', '', regex=False)
    df['monthly_charges'] = df['monthly_charges'].str.replace(' USD', '', regex=False)
    df['monthly_charges'] = df['monthly_charges'].str.replace(',', '', regex=False)
    df['monthly_charges'] = pd.to_numeric(df['monthly_charges'], errors='coerce')
    df.loc[~df['monthly_charges'].between(1, 300), 'monthly_charges'] = np.nan
    df['monthly_charges'].isnull().sum()

    return df

def support_calls_last_3_months(df):
    df['support_calls_last_3_months'] = df['support_calls_last_3_months'].astype(str).str.strip()
    df['support_calls_last_3_months'] = pd.to_numeric(
        df['support_calls_last_3_months'].str.replace(' many', '', regex=False),
        errors='coerce'
    )
    df.loc[~df['support_calls_last_3_months'].between(0, 10), 'support_calls_last_3_months'] = np.nan
    df['support_calls_last_3_months'].isnull().sum()

    return df

#6. Standardize categorical columns

def Standardize(df):
    categorical_columns = [
    'contract_type', 'internet_service', 'payment_method',
    'paperless_billing', 'has_dependents', 'tech_support'
]

    for col in categorical_columns:
        df[col] = df[col].astype('string').str.strip().str.lower()
        df[col] = df[col].replace({'': pd.NA, 'nan': pd.NA, 'none': pd.NA, 'n/a': pd.NA})

    df['contract_type'] = df['contract_type'].replace({
        'month to month': 'month-to-month', 'monthly': 'month-to-month',
        'one-year': 'one year', '1 year': 'one year',
        'two-year': 'two year', '2 year': 'two year', '2 years': 'two year'
    })

    df['internet_service'] = df['internet_service'].replace({
        'fiber': 'fiber optic', 'd.s.l.': 'dsl', 'none': 'no',
        'no internet': 'no', 'no internet service': 'no'
    })

    df['payment_method'] = df['payment_method'].replace({
        'e-check': 'electronic check', 'cc': 'credit card',
        'direct debit': 'bank transfer', 'paper check': 'mailed check'
    })

    for col in ['paperless_billing', 'has_dependents', 'tech_support']:
        df[col] = df[col].replace({'y': 'yes', 'n': 'no', 'true': 'yes', 'false': 'no', '1': 'yes', '0': 'no'})

    return df

#7. Clean target column churn

def target(df):
    df['churn'] = df['churn'].astype(str).str.strip().str.lower()
    df['churn'] = df['churn'].replace({
        'no': '0', 'not churned': '0', 'stayed': '0',
        'yes': '1', 'churned': '1', 'left': '1'
    })
    df['churn'] = pd.to_numeric(df['churn'], errors='coerce')
    df = df[df['churn'].isin([0, 1])].copy()
    df['churn'] = df['churn'].astype(int)
    return df 

def Remove(df):    
    df.drop(columns=['customer_id'], inplace=True)
    df.head()
    return df

def split(df):
    X = df.drop(columns=['churn'])
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

df = ingest_data("data/raw/messy_customer_churn_2000.csv")

df = drop_duplicated(df)
df = age_cleaning(df)
df = tenure_months(df)
df = monthly_charges(df)
df = support_calls_last_3_months(df)
df = Standardize(df)
df = target(df)
df = Remove(df)

X_train, X_test, y_train, y_test = split(df)

path_list = [X_train,X_test,y_train,y_test]

import os
output_path = os.path.join("data\processed")

X_train.to_csv(os.path.join(output_path , "X_train.csv"),index=False)
X_test.to_csv(os.path.join(output_path , "X_test.csv"),index=False)
y_train.to_csv(os.path.join(output_path , "y_train.csv"),index=False)
y_test.to_csv(os.path.join(output_path , "y_test.csv"),index=False)