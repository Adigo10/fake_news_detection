import pandas as pd 

def data_read(data_path):
    
    fake_df = pd.read_csv(data_path+"Fake.csv")
    real_df = pd.read_csv(data_path+"True.csv")

    fake_df = fake_df.iloc[:1500]

    real_df = real_df.iloc[:1500]
    
    print(fake_df.head())
    print(real_df.head())



    return fake_df,real_df