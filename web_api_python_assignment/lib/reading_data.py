import pandas as pd 

def data_read(fake_data,true_data):
    
    fake_df = pd.read_excel(fake_data)
    real_df = pd.read_excel(true_data)

    fake_df = fake_df.iloc[:1500]

    real_df = real_df.iloc[:1500]
    
    print(fake_df.head())
    print(real_df.head())



    return fake_df,real_df