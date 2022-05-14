# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:19:49 2022

@author: Administrator
"""

def Conc_model(X,P):
    X1 = X
    X1 = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2),padding='same')(X1)
    X1=BatchNormalization()(X1)
    X1=Activation('ELU')(X1)
    X1=Dropout(p)(X1,training=True)
    X1 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2),padding='same')(X1)
    X1=BatchNormalization()(X1)
    X1=Activation('ELU')(X1)
    X1=Dropout(p)(X1,training=True)
    X1 = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2),padding='same')(X1)
    X1=BatchNormalization()(X1)
    X1=Activation('ELU')(X1)
    X1=Dropout(p)(X1,training=True)
    X1 = Conv2D(filters=16, kernel_size=(3,3), strides=(2,2),padding='same')(X1)
    X1=BatchNormalization()(X1)
    X1=Activation('ELU')(X1)
    X1=Dropout(p)(X1,training=True)
    X1=Dense(512)(X1)
    X1=Activation('ELU')(X1)
    X1=Dropout(p)(X1,training=True)
    X1 = Flatten()(X1)
    
    X2 = X
    X2 = MaxPooling2D(pool_size=(2, 2))(X2)
    X2=BatchNormalization()(X2)
    X2=Activation('ELU')(X2)
    X2=Dropout(p)(X2,training=True)
    X2 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2),padding='same')(X2)
    X2=BatchNormalization()(X2)
    X2=Activation('ELU')(X2)
    X2=Dropout(p)(X2,training=True)
    X2 = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2),padding='same')(X2)
    X2=BatchNormalization()(X2)
    X2=Activation('ELU')(X2)
    X2=Dropout(p)(X2,training=True)
    X2 = Conv2D(filters=16, kernel_size=(3,3), strides=(2,2),padding='same')(X2)
    X2=BatchNormalization()(X2)
    X2=Activation('ELU')(X2)
    X2=Dropout(p)(X2,training=True)
    X2=Dense(512)(X2)
    X2=Activation('ELU')(X2)
    X2=Dropout(p)(X2,training=True)
    X2 = Flatten()(X2)

    X = Concatenate(axis=-1)([X1, X2])
    return X
