import pandas as pd 

def data_read(dataFake_path,dataReal_path):
    
    fake_df = pd.read_csv(dataFake_path)
    real_df = pd.read_csv(dataReal_path)

    print(fake_df.head())
    print(real_df.head())



    return fake_df,real_df