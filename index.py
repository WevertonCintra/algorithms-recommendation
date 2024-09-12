import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras
from pprint import pprint
from keras.utils import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# from keras.models import Model
# from keras.layers import Input, Reshape, Dot
# from keras.layers.embeddings import Embedding
# from keras.optimizers import Adam
# from keras.regularizers import l2
# from keras.layers import Concatenate, Dense, Dropout
# from keras.layers import Add, Activation, Lambda

DATASET_LINK='http://files.grouplens.org/datasets/movielens/ml-100k.zip'
!wget -nc http://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip -n ml-100k.zip

overall_stats = pd.read_csv('ml-100k/u.info', header=None)
print(list(overall_stats[0]))

## same item id is same as movie id, item id column is renamed as movie id
column_names1 = ['user id','movie id','rating','timestamp']
ratings_dataset = pd.read_csv('ml-100k/u.data', sep='\t',header=None,names=column_names1)
ratings_dataset.head()

column_names2 = [
    'movie id',
    'movie title',
    'release date',
    'video release date',
    'IMDb URL',
    'unknown',
    'Action',
    'Adventure',
    'Animation',
    'Children',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Fantasy',
    'Film-Noir',
    'Horror',
    'Musical',
    'Mystery',
    'Romance',
    'Sci-Fi',
    'Thriller',
    'War',
    'Western'
]

items_dataset = pd.read_csv('ml-100k/u.item', sep='|',header=None,names=column_names2,encoding='latin-1')
items_dataset.head()

movie_dataset = items_dataset[[
    'movie id',
    'release date',
    'Action',
    'Adventure',
    'Animation',
    'Children',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Fantasy',
    'Film-Noir',
    'Horror',
    'Musical',
    'Mystery',
    'Romance',
    'Sci-Fi',
    'Thriller',
    'War',
    'Western'
]]
movie_dataset.head()

movie_columns = [
    'Action',
    'Adventure',
    'Animation',
    'Children',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Fantasy',
    'Film-Noir',
    'Horror',
    'Musical',
    'Mystery',
    'Romance',
    'Sci-Fi',
    'Thriller',
    'War',
    'Western'
]
merged_dataset = pd.merge(ratings_dataset, movie_dataset, how='inner', on='movie id')
merged_dataset = merged_dataset[['user id', 'timestamp', 'rating'] + movie_columns]
merged_dataset.head()

movie_columns = [
    'Action',
    'Adventure',
    'Animation',
    'Children',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Fantasy',
    'Film-Noir',
    'Horror',
    'Musical',
    'Mystery',
    'Romance',
    'Sci-Fi',
    'Thriller',
    'War',
    'Western'
]
merged_dataset = pd.merge(ratings_dataset, movie_dataset, how='inner', on='movie id')
merged_dataset = merged_dataset[['user id', 'timestamp', 'rating'] + movie_columns]
merged_dataset.head()

# Extrair as sequências de features (gêneros e diferença de dias)
# feature_sequences = sequences['movieId_info'].apply(lambda x: [[movie[genre] for genre in movie_columns if genre != 'timestamp'] for movie in x])
feature_sequences = sequences['movieId_info'].apply(lambda x: [list(movie.values()) for movie in x])

# Encontrar o número máximo de filmes em uma sequência
max_sequence_length = sequences['movieId_info'].apply(len).max()
# max_sequence_length = 20

# Preencher as sequências com zeros para garantir que todas tenham o mesmo comprimento
padded_feature_sequences = pad_sequences(feature_sequences, maxlen=max_sequence_length, padding='pre', dtype='float32')

# 2. Preparar os dados para treinamento

# Separar as sequências em features (X) e labels (y)
X = padded_feature_sequences[:, :-1]  # Todas as etapas, exceto a última
y = padded_feature_sequences[:, -1, :-1]   # Última etapa (próximo filme)

# Dividir os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y[0]

print("Shape of X_train:", X_train.shape)

model = Sequential()
model.add(LSTM(80))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(len(movie_columns), activation='softmax'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=1000, validation_data=(X_test, y_test), callbacks=[early_stopping])

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model2 = Sequential()

model2.add(GRU(80))
model2.add(Dense(20, activation='relu'))
model2.add(Dense(20, activation='tanh'))
model2.add(Dense(20, activation='relu'))
model2.add(Dense(20, activation='relu'))
model2.add(Dense(len(movie_columns), activation='softmax'))

model2.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Treinar o modelo
history2 = model2.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# summarize history for accuracy
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

 summarize history for loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

