import pandas as pd 
import zipfile

def data_read(dataFake_path,dataReal_path):
    
    fake_df = pd.read_csv(dataFake_path)
    real_df = pd.read_csv(dataReal_path)

    print(fake_df.head())
    print(real_df.head())

    return fake_df,real_df

def file_read(dataFake_path):
    
    dataFrame = pd.read_csv(dataFake_path)
    return dataFrame


def unzip_data(request):
    inputZip = request.files['file']
    print(inputZip.filename)
    inputZip.save(inputZip.filename)  
    zipfile_ob = zipfile.ZipFile(inputZip)
    zipfile_ob.extractall('.')
    