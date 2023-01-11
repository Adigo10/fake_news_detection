from distutils.command.config import config
import pandas as pd


from src.reading_data import data_read
#from src.utility import set_logger, parse_config
from src.inference import evaluate
from src.plot import visualization_plot
from src.preprocessing import data_cleaning
from src.modeltraining import build_model
from src.tokenize import tokenize_data
import pickle
import mlflow
from mlflow.tracking import MlflowClient
import nltk
from sklearn.naive_bayes import MultinomialNB







def main():
    
    
    
    data_path = "data/"
    
    
    execution1 = data_read(data_path)
    
    
    execution2 = data_cleaning(execution1)
    
    
    
    X_train, X_test, y_train, y_test = tokenize_data(execution2)
    #mlflow_uri = "http://mlflow_tracker:5000"
    #mlflow.set_tracking_uri(mlflow_uri)
        
    EXPERIMENT_NAME = "som_NLP_assignment_temp_6"
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
    for idx,alpha in enumerate([0.1, 0.4, 0.8, 1]):


        #model,history = build_model(X_train,y_train)
        model = build_model(X_train,y_train,alpha)
            
            #pickled_model = pickle.load(open('src/model.pkl', 'rb'))
            #with open('model_pkl' , 'rb') as f:
            #model_received = pickle.load(f)
            
            #execution5 = visualization_plot(history)

            
        accuracy_score,precision_score,recall_score = evaluate(model,X_test,y_test)
        
        RUN_NAME = f"run_{idx}"
        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name = RUN_NAME) as run:
            RUN_ID = run.info.run_id
        
        #mlflow.set_experiment(experiment_name)
            mlflow.log_param("alpha", model.alpha)
            mlflow.log_param("model score",model.score)
            #mlflow.log_artifact(X_train,"xtrain repo")
            #mlflow.log_param("model_type", type(model))
            #mlflow.log_param("model_type", model.epochs())


            mlflow.log_metric("accuracy score ",accuracy_score)
            mlflow.log_metric("precision score ",precision_score)
            mlflow.log_metric("recall score ",recall_score)

            mlflow.sklearn.log_model(model, "classifier")


        
        
        #mlflow.log_param("n_layers", model.n_layers)
        #mlflow.log_param("n_hidden", model.n_hidden)
    


        #mlflow_uri = "http://mlflow_tracker:5000"
        #mlflow.set_tracking_uri(mlflow_uri)

        

       
    
    
    


    #return execution5


if __name__ == "__main__":
    main()