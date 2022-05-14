# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:16:03 2022

@author: Administrator
"""

def inter(X):
    X= X
    X=UpSampling2D(size=(3,3),interpolation='bilinear')(X)
    X = Conv2D(filters=64, kernel_size=(2, 2),strides=(1,1),
                 padding='same')(X)
    X = BatchNormalization()(X)
    X=Activation('ELU')(X)
    X=UpSampling2D(size=(3,3),interpolation='bilinear')(X)
    X = Conv2D(filters=32, kernel_size=(2, 2),strides=(1,1),
                 padding='same')(X)
    X = BatchNormalization()(X)
    X=Activation('ELU')(X)
    X=UpSampling2D(size=(3,3),interpolation='bilinear')(X)
    X = Conv2D(filters=16, kernel_size=(2, 2),strides=(1,1),
                 padding='same')(X)
    X = BatchNormalization()(X)
    X=Activation('ELU')(X)
    return X