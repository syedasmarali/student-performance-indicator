import pandas as pd
import os

def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Student_Performance.csv')
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    # Loading the data
    df = load_data()

    # Dropping the empty and none values
    df = df.dropna()

    # Binary encoding of extra curricular activities
    df = pd.get_dummies(df, columns=['Extracurricular Activities'], drop_first=True)

    # Print the dataset
    #pd.set_option('display.max_columns', None)
    #print(df.head())

    # Return df
    return df