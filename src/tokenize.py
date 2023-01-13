import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import re
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd





def tokenize_data(news_df):
    features = news_df['text']
    targets = news_df['class']

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.20, random_state=18)


    def normalize(data):
        normalized = []
        for i in data:
            i = i.lower()
            # get rid of urls
            i = re.sub('https?://\S+|www\.\S+', '', i)
            # get rid of non words and extra spaces
            i = re.sub('\\W', ' ', i)
            i = re.sub('\n', '', i)
            i = re.sub(' +', ' ', i)
            i = re.sub('^ ', '', i)
            i = re.sub(' $', '', i)
            normalized.append(i)
        return normalized

    X_train = normalize(X_train)
    X_test = normalize(X_test)
    
    
    
    
    
    max_vocab = 10000
    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=256)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=256)

    
    
    return X_train, X_test, y_train, y_test

def token_line(a):
    string = a
    df = pd.DataFrame([string], columns=['string_values'])
    x=df['string_values']

    def normalize(data):
        normalized = []
        for i in data:
            i = i.lower()
            # get rid of urls
            i = re.sub('https?://\S+|www\.\S+', '', i)
            # get rid of non words and extra spaces
            i = re.sub('\\W', ' ', i)
            i = re.sub('\n', '', i)
            i = re.sub(' +', ' ', i)
            i = re.sub('^ ', '', i)
            i = re.sub(' $', '', i)
            normalized.append(i)
        return normalized

    x=normalize(x)
    max_vocab = 10000
    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(x)

    x = tokenizer.texts_to_sequences(x)

    x= tf.keras.preprocessing.sequence.pad_sequences(x, padding='post', maxlen=257)

    return x

def token_file(data):
    x=data['text']
    
    def normalize(data):
        normalized = []
        for i in data:
            i = i.lower()
            # get rid of urls
            i = re.sub('https?://\S+|www\.\S+', '', i)
            # get rid of non words and extra spaces
            i = re.sub('\\W', ' ', i)
            i = re.sub('\n', '', i)
            i = re.sub(' +', ' ', i)
            i = re.sub('^ ', '', i)
            i = re.sub(' $', '', i)
            normalized.append(i)
        return normalized

    x=normalize(x)
    max_vocab = 10000
    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(x)

    x = tokenizer.texts_to_sequences(x)

    x= tf.keras.preprocessing.sequence.pad_sequences(x, padding='post', maxlen=257)

    return x