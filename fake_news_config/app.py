from flask import Flask,render_template,url_for,request,jsonify, make_response, redirect, request
import pandas as pd
import numpy as np

from src.reading_data import data_read,file_read
from src.reading_data import unzip_data
#from src.utility import set_logger, parse_config
from src.inference import evaluate
from src.plot import visualization_plot
from src.preprocessing import data_cleaning
from src.modeltraining import build_model
from src.tokenize import tokenize_data
from sklearn.feature_extraction.text import CountVectorizer
import mlflow
import pickle

import sys
import os
import configparser
import logging
import logging.config

app = Flask(__name__)
config = None
config = configparser.ConfigParser(allow_no_value=True)
config._interpolation = configparser.ExtendedInterpolation()
config.read('Config.ini')
#print(config)

@app.route('/preprocess',methods=['POST'])
def preprocess():
    #unzip file object  
    unzip_data(request)

    #read data
    dataFakePath = config['Default']['Fake_path']
    dataTruePath = config['Default']['True_path']
    data = data_read(dataFakePath,dataTruePath)
    
    #data cleaning
    cleanData = data_cleaning(data)
    
    tokenFilePath = config['Default']['tokenizer_file']
    
    # tokenize data
    X_train, X_test, y_train, y_test = tokenize_data(cleanData,tokenFilePath)
    
    
    
    X_train_df = pd.DataFrame(X_train) 
    xTrainPath = config['Default']['x_train']
    X_train_df.to_csv(xTrainPath)
    
    X_test_df = pd.DataFrame(X_test) 
    xTestPath = config['Default']['x_test']
    X_test_df.to_csv(xTestPath)
    
    y_train_df = pd.DataFrame(y_train) 
    yTrainPath = config['Default']['y_train']
    y_train_df.to_csv(yTrainPath)
    
    y_test_df = pd.DataFrame(y_test) 
    yTestPath = config['Default']['y_test']
    y_test_df.to_csv(yTestPath)
    
    
    response = make_response(jsonify({"message": "YAHOOOO!!", "severity": "delightful"}),200,)
    response.headers["Content-Type"] = "application/json"
    return response

@app.route('/train',methods=['POST'])
def train():
    EXPERIMENT_NAME = request.args.get('experimentname')
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
    
    # read X_train, X_test, y_train, y_test from csv
    xTrainPath = config['Default']['x_train']
    X_train_df = pd.read_csv(xTrainPath)
    X_train = X_train_df.to_numpy()
    
    xTestPath = config['Default']['x_test']
    X_test_df = pd.read_csv(xTestPath)
    X_test = X_test_df.to_numpy()
    
    yTrainPath = config['Default']['y_train']
    y_train_df = pd.read_csv(yTrainPath,usecols=["class"])
    y_train = y_train_df.to_numpy()
    
    yTestPath = config['Default']['y_test']
    y_test_df = pd.read_csv(yTestPath,usecols=["class"])
    y_test = y_test_df.to_numpy()
    
    alphaValues = config['Default']['alpha_values']
    pickleFilePath = config['Default']['pickel_file']
    for idx, alpha in enumerate(alphaValues):
        
        model = build_model(X_train,y_train,alpha)
        pickle.dump(model, open(pickleFilePath, 'wb'))
  
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


@app.route('/predict',methods=['POST'])
def predict():
    EXPERIMENT_NAME = request.args.get('experimentname')
    EXPERIMENT_ID = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    
    # read X_train, X_test, y_train, y_test from csv
    xTestPath = config['Default']['x_test']
    X_test_df = pd.read_csv(xTestPath)
    X_test = X_test_df.to_numpy()
    
    yTestPath = config['Default']['y_test']
    y_test_df = pd.read_csv(yTestPath,usecols=["class"])
    y_test = y_test_df.to_numpy()

    file_path = request.args.get('file_path')
    data = file_read(file_path)
    
    tokenFilePath = config['Default']['tokenizer_file']
    
    pickled_model = config['Default']['pickel_file']
    baseModel = pickle.load(open(pickled_model, 'rb'))
    #data cleaning
    tokenData = token_file(data,tokenFilePath)

    data['result'] = baseModel.predict(tokenData)

    resultFN = config['Default']['result_file']

    outdir = config['Default']['result_path']

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fullname = os.path.join(outdir, outname)    

    data.to_csv(fullname)

    resultDF=baseModel.predict(tokenData)

    response = make_response(jsonify({"message": "YAHOOOO!!", "severity": "delightful"}),200,)
    response.headers["Content-Type"] = "application/json"
    return response
    
if __name__ == '__main__':
   #config = configparser.ConfigParser(allow_no_value=True)
   #config._interpolation = configparser.ExtendedInterpolation()
   #config.read('Config.ini')
   #print(config)# Read our Config File
   app.run(host='0.0.0.0',port=5001)
    
    
    
    