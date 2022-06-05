# PJP-Velocity-Field-Prediction
It uses a variational Bayesian neural network to predict the velocity field of the PJP.
code description

1.It takes discrete pressure point data as input to the model and velocity field data as output. Only three sets of data for velocity field data and discrete pressure points are provided here. 
2.The code “Norscaler” is used to normalize the dataset.
3.The code “upsampling” will process the discrete pressure point data to a resolution corresponding to the velocity field data.
4.The code “model” and “CNN_VB” is the main part of the multipath model.
5.The code “evalute_VB” is used for uncertainty analysis.

When the model is trained, the optimal weights and biases will be saved. In practical engineering, it only needs to provide discrete pressure point data, and it can obtain the corresponding velocity field data.
The output data of the training data-set has exceeded 1GB, which has exceeded the capacity limit. If necessary, it can be sent by e-mail.(qiuchengcheng1010@163.com)
