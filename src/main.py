from data_processing import load_data, preprocess_data

def main():
    # Load the dataset
    df = load_data()

    # Preprocess the data
    df_cleaned = preprocess_data(df)

if __name__ == '__main__':
    main()