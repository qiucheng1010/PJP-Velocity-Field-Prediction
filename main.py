# -*- coding: utf-8 -*-
"""
Created on Sat May 14 08:25:24 2022

@author: Administrator
"""

def main_run(dataset, verbose, epochs_num, batch_size,sample_times,z_value,p):
    model,history = CNN_VB_model(dataset, verbose_set, epochs_num, batch_size,p)
    print('\ntrain_acc:%s'%np.mean(history.history['accuracy']), '\ntrain_loss:%s'%np.mean(history.history['loss']))
    loss, accuracy= model.evaluate(test_x,test_y, verbose=1)

    yhat= model.predict(test_x, verbose=1
    
    upper_bound, lower_bound,mean_y, var_y= evalute_mc(model, test_x,test_y,batch_size,sample_times,z_value,p)
  