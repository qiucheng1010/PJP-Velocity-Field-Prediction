# -*- coding: utf-8 -*-
"""
Created on Sat May 14 08:21:24 2022

@author: Administrator
"""

def CNN_VB_model(dataset, verbose_set, eporchs_num, bacth_size,p):
    train_x,train_y,test_x,test_y=dataset
    
    S_inputs=Input(shape=(train_x.shape[1:]))
    X=inter(S_inputs)
    X=Conv_model(X,p)
    outputs=Reshape((train_y.shape[1:]))(X)
    model=Model(inputs=S_inputs, outputs=outputs)         
    lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LR,
    decay_steps=DS,
    decay_rate=0.9)	
    optimizer_decay = optimizers.Adam(learning_rate=lr_schedule)  
    model.compile(loss='mse', optimizer = optimizer_decay)
    print(model.summary())
    history=model.fit(train_x, train_y, verbose=verbose_set, epochs=epochs_num, batch_size=batch_size,
                      validation_data=(test_x, test_y), validation_freq=1,
                      callbacks=[TensorBoard(log_dir='1.log', histogram_freq=1)])
    return model,history
