import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import re 
import pickle


def build_model(X_train,y_train,alpha):
    mnv = MultinomialNB(alpha)
    model = mnv.fit(X_train,y_train)
    pickle.dump(model, open('model.pkl', 'wb'))
    return model
#    max_vocab=10000
#    model = tf.keras.Sequential([
#    tf.keras.layers.Embedding(max_vocab, 128),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
#    tf.keras.layers.Dense(64, activation='relu'),
#    tf.keras.layers.Dropout(0.5),
#    tf.keras.layers.Dense(1)])
#
#
#    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
#    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#              optimizer=tf.keras.optimizers.Adam(1e-4),
#              metrics=['accuracy'])
#
#    
#    #pickle.dump(model, open('model.pkl', 'wb'))
#    
#    history = model.fit(X_train, y_train, epochs=1,validation_split=0.1, batch_size=batchSize, shuffle=True, callbacks=[early_stop])
#
#    return model,history

    
