import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

""""
The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. 
The classification goal is to predict if the client will subscribe a term deposit (variable y).

"""

def load_bank_marketing_dataset(): # Function to load data
    bank_marketing = fetch_openml(name='bank-marketing')
    df = pd.DataFrame(bank_marketing.data, columns=bank_marketing.feature_names)
    df['y'] = bank_marketing.target  # Target column
    df.drop(columns=['V12'], inplace=True) # Removing V12 feature as suggested by the UCI: https://archive.ics.uci.edu/dataset/222/bank+marketing
    return df


def preprocess_data(df):
    """
    1. One-hot encoding for categorical features
     a. Convert to int (one-hot encoding dtype: Category)
    2. Label encoding for binary features
    3. Minmax Scale for integers 
    """
    # One-hot encoding for categorical features
    df = pd.get_dummies(df, columns=['V2','V3','V4','V9','V11','V16'])
    # get_dummies creates TRUE FALSE. Convert to int otherwise dtype: Category
    for column in df.columns:
    # Check if the first element in the column is a boolean
        if pd.api.types.is_bool_dtype(df[column].iloc[0]):
            # Map TRUE to 1 and FALSE to 0
            df[column] = df[column].astype(int)
            
    # Label encoding for binary features
    label_map = {'yes': 1, 'no': 0} # Unique values of categories 1, 0
    #Convert to int, otherwise dtype: Category
    df['V5'] = df['V5'].map(label_map).astype(int)
    df['V7'] = df['V7'].map(label_map).astype(int)
    df['V8'] = df['V8'].map(label_map).astype(int)
    label_map={'1' : 0, '2': 1}     # Unique values of y : 1, 2
    df['y']=df['y'].map(label_map).astype(int)

    # Min-max scaling for numeric features
    scaler = MinMaxScaler()
    numeric_columns = ['V6', 'V13', 'V14', 'V15'] 
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


def optimize_memory_usage(df):
    """
    Optimizing memory usage by downcasting integer and floating-point columns accordingly.
    """
    for column in df.columns:
        # Downcast integer-type columns
        if np.issubdtype(df[column].dtype, np.integer):
            max_value = df[column].max()
            if max_value < np.iinfo(np.int8).max:
                df[column] = df[column].astype(np.int8)
            elif max_value < np.iinfo(np.int16).max:
                df[column] = df[column].astype(np.int16)
            elif max_value < np.iinfo(np.int32).max:
                df[column] = df[column].astype(np.int32)
            else:
                df[column] = df[column].astype(np.int64)
        # Downcast floating-point columns
        elif np.issubdtype(df[column].dtype, np.floating):
            max_value = df[column].max()
            min_value = df[column].min()
            if (min_value > np.finfo(np.float32).min and
                    max_value < np.finfo(np.float32).max):
                df[column] = df[column].astype(np.float32)
            else:
                # If the values are outside the range of float32,
                # retain float64 to avoid losing precision
                pass
    return df

bank_data = load_bank_marketing_dataset() # Load dataset
df = preprocess_data(bank_data) # def Process data 
df = optimize_memory_usage(df) # def Optimize Memory Usage
df.to_csv('Bank_data.csv', index=False)