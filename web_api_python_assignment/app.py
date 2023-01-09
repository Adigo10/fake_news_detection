from flask import Flask, request, render_template
#from werkzeug import secure_filename
import pandas as pd
#from pre_processed import data_clean
from lib.reading_data import data_read
from lib.preprocessing import data_cleaning
import pickle

#render_template will redirtect to the first home page
app = Flask(__name__)

@app.route("/")
def upload():
    return render_template('upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        fake_data = request.files['fake_data']
        true_data = request.files['true_data']

        #data_fake = pd.read_excel(f)
        limited_records  = data_read(fake_data,true_data)
        print(limited_records)

        data_mining = data_cleaning(limited_records)


        
        
        #data = request.form['sample_text']
        
        #return render_template('upload.html', data_mining)
        return data_mining.head().to_html()
        #return df
    
#def preprocess():
    #return "soumya"
'''
@app.route('/pre-process', methods = ['GET','POST'])
def pre_process():
    if request.method == 'POST':
        df_fake  = pickle.load(open('data.pkl', 'rb'))
        
        pre_processed_df = data_clean(df_fake)
        
        return df_fake.head().to_html()
        
@app.roaute('/tokenize',methods = ['GET','POST'])
def tokenize
'''


if __name__ == '__main__':
    app.run(debug=True)
    