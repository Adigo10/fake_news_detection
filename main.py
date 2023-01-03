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



def main():

    
    dataFake_path = "./data/Fake.csv"
    dataReal_path = "./data/True.csv"
    execution1 = data_read(dataFake_path,dataReal_path)
    execution2 = data_cleaning(execution1)
    
    X_train, X_test, y_train, y_test = tokenize_data(execution2)
    
    model,history = build_model(X_train,y_train)
    ##check if we can return model
    #pickled_model = pickle.load(open('model.pkl', 'rb'))
    
    execution5 = visualization_plot(history)

    execution6 = evaluate(model,X_train, X_test, y_train, y_test)
    
    #execution5 = inference(execution4)


    return execution5


if __name__ == "__main__":
    main()