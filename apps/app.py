from flask import Flask,render_template,url_for,request,jsonify, make_response, redirect, request
import pandas as pd
import numpy as np
import os
from src.reading_data import data_read
from src.reading_data import file_read
#from src.utility import set_logger, parse_config
from src.inference import evaluate
from src.plot import visualization_plot
from src.preprocessing import data_cleaning
from src.modeltraining import build_model
from src.tokenize import tokenize_data
from src.tokenize import token_line
from src.tokenize import token_file
from sklearn.feature_extraction.text import CountVectorizer
import mlflow
import pickle

app = Flask(__name__)

#@app.route('/'.methods=['GET','POST'])

@app.route('/preprocess',methods=['GET','POST'])
def preprocess():
    #read data 
    # dataFakePath = r'C:\Users\callm\all\fake_news_detection-master\data'
    dataTruePath = request.args.get('fakepath')
    data = data_read(dataTruePath)
    
    #data cleaning
    cleanData = data_cleaning(data)
    
    # tokenize data
    X_train, X_test, y_train, y_test = tokenize_data(cleanData)
    
    X_train_df = pd.DataFrame(X_train) 
    X_train_df.to_csv('X_train.csv')
    
    X_test_df = pd.DataFrame(X_test) 
    X_test_df.to_csv('X_test.csv')
    
    y_train_df = pd.DataFrame(y_train) 
    y_train_df.to_csv('y_train.csv')
    
    y_test_df = pd.DataFrame(y_test) 
    y_test_df.to_csv('y_test.csv')
    
    
    response = make_response(jsonify({"message": "YAHOOOO!!", "severity": "delightful"}),200,)
    response.headers["Content-Type"] = "application/json"
    return response


@app.route('/predict',methods=['GET','POST'])
def predict():
    EXPERIMENT_NAME = request.args.get('experimentname')
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
    
    # read X_train, X_test, y_train, y_test from csv
    X_train_df = pd.read_csv('X_train.csv')
    X_train = X_train_df.to_numpy()
    print(np.shape(X_train)) 
    
    X_test_df = pd.read_csv('X_test.csv')
    X_test = X_test_df.to_numpy()
    
    y_train_df = pd.read_csv('y_train.csv',usecols=["class"])
    y_train = y_train_df.to_numpy()
    print(np.shape(y_train)) 
    
    y_test_df = pd.read_csv('y_test.csv',usecols=["class"])
    y_test = y_test_df.to_numpy()
    
    for idx, alpha in enumerate([0.2, 0.3, 0.4]):
        
        model = build_model(X_train,y_train,alpha)
        pickle.dump(model, open('model.pkl', 'wb'))
        
        accuracy,presision,recall = evaluate(model,X_test, y_test)
        
        ##check if we can return model
        #pickled_model = pickle.load(open('model.pkl', 'rb'))
    
        # Start MLflow
        RUN_NAME = f"run_{idx}"
        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME) as run:
            # Retrieve run id
            RUN_ID = run.info.run_id

            #visualization_plot(model,mlflow) # logging plots in ML Flow
            # Track parameters
            mlflow.log_param("alpha", alpha)
            
            # Track parameters
            mlflow.log_param("score", model.score)

            # Track metrics
            mlflow.log_metric("accuracy", accuracy)
            
            # Track metrics
            mlflow.log_metric("presision", presision)
            
            # Track metrics
            mlflow.log_metric("recall", recall)
            
            # Track metrics
            #mlflow.log_metric("prediction", prediction)

            # Track model
            mlflow.sklearn.log_model(model, "model")

    response = make_response(jsonify({"message": "YAHOOOO!!", "severity": "delightful"}),200,)
    response.headers["Content-Type"] = "application/json"
    return response

@app.route('/text',methods=['GET','POST'])
def prediction():
    text = request.args.get('text')
    print(text)
    joblib_model = pickle.load(open('model.pkl', 'rb'))
    y=token_line(text)
    z=joblib_model.predict(y)
    print("#############")
    print(z)
    print("########")

    response = make_response(jsonify({"message": "YAHOOOO!!", "severity": "delightful"}),200,)
    response.headers["Content-Type"] = "application/json"
    return response

@app.route('/file',methods=['GET','POST'])
def prediction_file():
    file_path = request.args.get('file_path')
    data = file_read(file_path)
    print('data type is:',type(data))
    joblib_model = pickle.load(open('model.pkl', 'rb'))
    y=token_file(data)
    z=joblib_model.predict(y)
    print("RESULT TIME :#############")
    print(z)
    print("########")

    data['result'] = z

    outname = 'result.csv'

    outdir = r'C:\Users\callm\all\fake_news_detection-master\apps'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fullname = os.path.join(outdir, outname)    

    data.to_csv(fullname)

    response = make_response(jsonify({"message": "YAHOOOO!!", "severity": "delightful"}),200,)
    response.headers["Content-Type"] = "application/json"
    return response
    
if __name__ == '__main__':
	app.run(host='0.0.0.0',port=80)
    
    
    
    