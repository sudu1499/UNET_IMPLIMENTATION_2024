from keras import layers,Model
from design_model import design_model
import json
import pickle as pkl
from sklearn.model_selection import train_test_split
import numpy as np

model=design_model()

config=json.load(open("config.json","r"))
x=pkl.load(open(config['data_x'],"rb"))
y=pkl.load(open(config['data_y'],"rb"))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=42)

from data_preperation import display_img,main_call

#main_call()
#display_img(x_train[0],y_train[0],5000)
x_train=np.divide(x_train,255)
x_test=np.divide(x_test,255)
y_train=np.divide(y_train,255)
y_test=np.divide(y_test,255)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuracy')
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=16)

model.predict(x_test)
