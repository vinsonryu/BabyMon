import pandas as pd
import os
import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
import sklearn
import tensorflow
from tqdm import tqdm

# test code---------------------------------------------------------------
file_name='cryingbaby/1-187207-A.ogg'
audio_data, sampling_rate = librosa.load(file_name)
plt.figure(figsize=(12, 5))
librosa.display.waveshow(audio_data,sr=sampling_rate)
ipd.Audio(file_name)
print(sampling_rate,audio_data)
mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=40)
print(mfccs.shape)
print(mfccs)
# test code---------------------------------------------------------------


#features extraction getting spectogram of audio
def features_extractor(file):
    #load the file (audio)
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    #we extract mfcc
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    #in order to find out scaled feature we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

# ## Now we iterate through every audio file and extract features 
# extracted_features=[]
for file_name in os.listdir('cryingbaby'):
    f = f'cryingbaby/{file_name}'
    # print(f)
    final_class_labels="cryingbaby"
    data=features_extractor(f)
    extracted_features.append([data,final_class_labels])
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()

# ### Split the dataset into independent and dependent dataset
# X=np.array(extracted_features_df['feature'].tolist())
# y=np.array(extracted_features_df['class'].tolist())
# ### Label Encoding -> Label Encoder
# from tensorflow.keras.utils import to_categorical
# from sklearn.preprocessing import LabelEncoder
# labelencoder=LabelEncoder()
# y=to_categorical(labelencoder.fit_transform(y))
# ### Train Test Split
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
# from tensorflow.keras.optimizers import Adam
# from sklearn import metrics
# ### No of classes
# num_labels=y.shape[1]
# model=Sequential()
# ###first layer
# model.add(Dense(100,input_shape=(40,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# ###second layer
# model.add(Dense(200))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# ###third layer
# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# ###final layer
# model.add(Dense(num_labels))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

# ## Trianing my model
# from tensorflow.keras.callbacks import ModelCheckpoint
# from datetime import datetime 
# num_epochs = 100
# num_batch_size = 32
# checkpointer = ModelCheckpoint(filepath='./audio_classification.hdf5', 
#                                verbose=1, save_best_only=True)
# start = datetime.now()
# model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
# duration = datetime.now() - start
# print("Training completed in time: ", duration)

# filename="harvard.wav"
# #preprocess the audio file
# audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
# mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
# mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
# #Reshape MFCC feature to 2-D array
# mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
# #predicted_label=model.predict_classes(mfccs_scaled_features)
# x_predict=model.predict(mfccs_scaled_features) 
# predicted_label=np.argmax(x_predict,axis=1)
# print(predicted_label)
# prediction_class = labelencoder.inverse_transform(predicted_label) 
# print(prediction_class)