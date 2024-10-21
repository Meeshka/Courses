import csv
import torch
import pandas as pd
import torch.utils.data as data_utils

def generate_tensor(df):
    pd.to_numeric(df["sessions"])
    sessions_df = pd.DataFrame(df['sessions'])

    # creating tensor from session_df
    return torch.tensor(sessions_df['sessions'].values)




if __name__ == "__main__":
    Data_path = "Tutors.csv"
    df = pd.read_csv(Data_path,nrows=50)
    print(df.head())
    train_tensor = generate_tensor(df)
    print(train_tensor)
