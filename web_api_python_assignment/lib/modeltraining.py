import tensorflow as tf
#from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import re 
import pickle
import mlflow
from mlflow.tracking import MlflowClient

max_vocab = 10000
def build_model(X_train,y_train):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_vocab, 128),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

    #pickle.dump(model, open('src/model.pkl', 'wb'))
    #with open('model_pkl', 'wb') as file:
    #pickle.dump(model, file)
    
    
    
    print(len(X_train))
    print(len(y_train))
    
    history = model.fit(X_train, y_train, epochs=5,validation_split=0.1, batch_size=100, shuffle=True, callbacks=[early_stop])
    
    return history,model


    
