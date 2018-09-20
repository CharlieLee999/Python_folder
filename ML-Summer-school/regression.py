#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:59:10 2018

@author: charlie
"""
import copy 
import os
import time
import pandas as pd
import numpy as np
#from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras import regularizers
#from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
### Parameters of the Neural Network ###
num_hidden_layers = 2
num_nodes = [66, 33, 40, 40]
activ_func = ['relu', 'tanh', 'relu', 'relu']
batch_size = 6
num_test = 20
epochs = 1000
epochs_step = epochs
ratio_test = 0.2

def baseline_model(num_nodes, activ_func):
    # create model
    model = Sequential()
    
    model.add(Dense(num_nodes[0], input_dim=6, kernel_initializer='uniform', activation=activ_func[0]))#, kernel_regularizer=regularizers.l2(0.05))) 
    
#    model.add(Dropout(0.2)) 
    
    model.add(Dense(num_nodes[1], kernel_initializer='uniform', activation=activ_func[1]))
    # kernel_regularizer=regularizers.l2(0.01)))
    
#    model.add(Dropout(0.2))
#    
    model.add(Dense(num_nodes[2], kernel_initializer='normal', activation=activ_func[2]))
    
#    model.add(Dense(num_nodes[3], kernel_initializer='normal', activation=activ_func[3]))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.add(Dense(1, kernel_initializer='normal')) # Default-> linear
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['mse'])
    return model

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

'''
index c mn v si cr mo ni w page bs-exp other bs 
0     1  2 3  4  5 6  7  8  9   10      
'''

fname = "DataBainite_summarizingDataBS.csv"
data = pd.read_csv(fname, sep=',', header = None)
corr_mat = data.corr()


xColumn = 1
yColumn = 2
rowList = [0, 1, 3, 4, 5, 9, 10, 15, 17, 21, 24, 25, 27, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 46, 47, 55, 63, 67, 74, 76]

#indexList = [0, 1, 3, 4, 5, 9, 10, 15, 17, 21, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
#             43, 44, 45, 46, 47, 48, 49, 50, 55, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78]
xData = data.values[rowList, :]

yData = data.values[rowList, 10]

xData = xData[:, [1, 2, 4, 5, 6, 7]]
#x_corr = xData.corr()
#xData = data.values[indexList, [1, 2, 5, 6, 7]]
scaler = StandardScaler()
scaler.fit(xData)
xData = scaler.transform(xData)

model = baseline_model(num_nodes, activ_func)

### weight folder ###
weight_dir = "model_weight_regre"
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

hist_dict = {}
for iter_test in range(num_test):
    curr_epoch = 0
    
    X_train, X_test, Y_train, Y_test = train_test_split(xData, yData, test_size= ratio_test)
    st_time = time.time()
    while( curr_epoch < epochs):
        print("\nCurrent test-> "+ str(iter_test)+ ", current epoch-> " + str(curr_epoch))
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs_step, verbose=0, validation_data=(X_test, Y_test))
        curr_epoch += epochs_step
    
    
    end_time = time.time()
    print('Test num-> ' + str(iter_test) + " timecost->" + str(round(end_time-st_time, 2)) )   
    hist_dict[iter_test] = history
    print("Current min training loss-> " + str(min(history.history['loss'])) )
    print("Current min test loss-> " + str(min(history.history['val_loss'])) )
    
    ### Save weight ###
    model.save_weights(weight_dir + '/weight_test_num_'+str(iter_test)+".h5")
    
    
    if(iter_test< num_test-1):
#        pass
        reset_weights(model)  ## Reset weights
#    print("Current min training error-> " + str(min(history.history['mean_absolute_error'])) )
#    print("Current min test error-> " + str(min(history.history['val_mean_absolute_error'])) )


acc_loss_dir = "accuracy_loss_regression"
if not os.path.exists(acc_loss_dir):
    os.makedirs(acc_loss_dir)

for i in range(len(hist_dict)):
    history = hist_dict[i]
    
    if (i==0):
#        err_train_aver = np.zeros(epochs)
#        err_valid_aver = np.zeros(epochs)
        loss_train_aver = np.zeros(epochs)
        loss_valid_aver = np.zeros(epochs)
    
#    err_train_aver  += np.array(history.history['mean_absolute_error'])
#    err_valid_aver  += np.array(history.history['val_mean_absolute_error'])
    loss_train_aver += np.array(history.history['loss'])
    loss_valid_aver += np.array(history.history['val_loss'])

### Get average of accuracy and loss ###
#acc_train_aver /= float(num_test)
#acc_valid_aver /= float(num_test)
loss_train_aver /= float(num_test)
loss_valid_aver /= float(num_test)

epochs_array = np.arange(epochs)

#plt.figure()
#plt.plot(epochs_array, err_train_aver)
#plt.plot(epochs_array, err_valid_aver)
#plt.title('model mean absolute error (average) ')
#plt.ylabel('Mean absolute error')
#plt.xlabel('epoch')
#plt.grid()
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig(acc_loss_dir+"/accuracy_aver.pdf", format='pdf')


plt.figure()
plt.plot(epochs_array, loss_train_aver)
plt.plot(epochs_array, loss_valid_aver)
plt.title('model mean absolute error (average) ')
plt.ylabel('mean absolute error')
plt.xlabel('epoch')
#plt.ylim([0, 2000])
plt.grid()
plt.legend(['train', 'test'], loc='upper right')
plt.savefig(acc_loss_dir+'/mae_' + activ_func[0]+"_nodes_"+str(num_nodes[0])+'.pdf')

yData_pred = np.zeros(len(rowList)).reshape(33,1)
for i in range(num_test):
    model.load_weights(weight_dir + '/weight_test_num_'+str(i)+".h5")
    yData_pred += model.predict(xData)
yData_pred /= float(num_test)


plt.figure()
plt.plot(yData, label='Exact y')
plt.plot(yData_pred, label='Pred y')
plt.legend()
plt.ylabel('Y')
plt.title('Exact y - predicted y (average)')
plt.savefig(acc_loss_dir+'/exact_pred_y.pdf', format='pdf')
#plt.savefig(acc_loss_dir+"/loss_aver_regr_"+ activ_func[0]+"_nodes_"+str(num_nodes[0])+".pdf", format='pdf')

### K-Fold validation ###
#seed = 7
#np.random.seed(seed)
## evaluate model with standardized dataset
#estimator = KerasRegressor(build_fn=baselines_model, epochs=100, batch_size=5, verbose=0)
#
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(estimator, xData, yData, cv=kfold)
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


#for i in range(len(data.values)):
#    if data.values[i, 5] != 0 and data.values[i, 6] != 0 and data.values[i, 7] != 0:
#        print(i)

#data_shape = data.shape
#
## num_nonzero_column = np.count_nonzero(data.values, axis=0)
#
## num_zero_column = data_shape[0] - num_nonzero_column
#
#regr = linear_model.LinearRegression()
#regr.fit(xData, yData)
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#X, Y = np.meshgrid(xData[:, 0], xData[:, 1])
#
#ax.scatter(xData[:, 0], xData[:, 1], yData)
#
#xlim = ax.get_xlim()
#ylim = ax.get_ylim()
#
#pred = regr.predict(np.array([[xlim[0], ylim[0]], [xlim[1], ylim[0]], [xlim[0], ylim[1]], [xlim[1], ylim[1]]])).reshape(2, 2)
#print(regr.coef_)
#XPlot, YPlot = np.meshgrid([xlim[0], xlim[1]], [ylim[0], ylim[1]])
#
#labelDict = {1: "C", 2: "Mn", 3: "V", 4: "Si", 5: "Cr", 6: "Mo", 7: "Ni", 8: "W"}
#
#ax.plot_surface(XPlot, YPlot, pred, alpha=.3)
#ax.set_xlabel(labelDict[xColumn])
#ax.set_ylabel(labelDict[yColumn])
#ax.set_zlabel("T in [K]")
#plt.show()

