# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:34:42 2022

@author: Administrator
"""

def evalute_mc(model, x_test, y_test,batch_size,sample_times,z_value,p):
    batch_size=batch_size

    for batch_id in tqdm(range(x_test.shape[0] // batch_size)):
     
        x = x_test[batch_id * batch_size: (batch_id + 1) * batch_size]
       
        y_ = np.zeros((sample_times, batch_size, y_test[0].shape[0],y_test[0].shape[1],y_test[0].shape[2]), dtype=np.float32)
        for sample_id in range(sample_times): 
            y_[sample_id] = model.predict(x)
        mean_y = y_.mean(axis=0)
        var_y=y_.var(axis=0)
        std_y=y_.std(axis=0)
        upper_bound=mean_y+z_value*(std_y)
        lower_bound=mean_y-z_value*(std_y)

    return upper_bound, lower_bound,mean_y,var_y
