#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 20:11:37 2018

@author: charlie
"""
import copy
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras import backend as K
from keras.optimizers import RMSprop, Adadelta, Adam, SGD
from ann_visualizer.visualize import ann_viz

### Ratio of testing data in the whole data
ratio_test = 0.2

### Parameters of training the Neural Network
activ_func = ["linear", "tanh", "tanh"]
batch_size = 6
epochs= 2000
epochs_step = int(epochs/50)
num_test = 10

flag_plot_tmp_contour = False
flag_normlize = True
flag_add_full_one = False

fname_complex = 'ANN_data_poisoning_products.txt'
fname_simple = 'dataset.txt'
contour_dir = "contour"

def getANN(dim_input, activ_func = ["linear","linear", "relu"]):
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    ### Default-> K.epsilon() -> 1e-07
    model = Sequential()
    
    ### First hidden layer ### #
    model.add(Dense(3*dim_input, input_dim = dim_input, activation=activ_func[0], kernel_initializer='glorot_uniform'))
    ### default kernel initializer -> glorot_uniform
#    model.add(Dropout(0.2))
    
    ## Second hidden layer ###
#    model.add(Dense(2*dim_input, activation=activ_func[1], kernel_initializer='glorot_uniform'))
##    model.add(BatchNormalization())
##    model.add(Activation(activ_func[1]))
###    
#    
#    ### Third hidden layer ###
#    model.add(Dense(3*dim_input, activation=activ_func[2], kernel_initializer='glorot_uniform'))
#    model.add(BatchNormalization())
#    model.add(Activation(activ_func[2]))
#    
    
    ### Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    
    st_time = time.time()
    model.compile(loss='binary_crossentropy', optimizer = sgd, metrics=['accuracy'])
    end_time = time.time()
    print("\n\n\n*****Neural Network compiling timecost-> " + str(end_time-st_time))
    
    ### Visualize the structure of Neural Network 
#    ann_viz(model, view=True, filename="FCNN_three_hidden_layers.pdf", title="Neural Network with three hidden layers")
    
    return model

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

def getLabelColorInGrid(X_orig, scaler, model, flag_normlize=True, flag_add_full_one=True):
    ### Get the range of x0 and x1 ###
    x0_min = min(X_orig[:,0]) - 1.0
    x0_max = max(X_orig[:,0]) + 1.0 # 10
    x1_min = min(X_orig[:,1]) 
    x1_max = max(X_orig[:,1]) # 12
    
    ### Get the disrete grid ###
#    x0_step_num = int((x0_max- x0_min)/0.3)
#    x1_step_num = int((x1_max-x1_min)/0.3)
    x0_step_num = 200
    x1_step_num = 200
    x0_coor = np.linspace(x0_min, x0_max, x0_step_num+1, endpoint=True)
    
    x1_coor = np.linspace(x1_min, x1_max, x1_step_num+1, endpoint=True)
    
    ### Get the grid 
    x0_grid, x1_grid = np.meshgrid(x0_coor, x1_coor)
    
    ### Reshape data and get X_simulated
    x0_grid_shape = x0_grid.shape
    grid_size = x0_grid_shape[0] * x0_grid_shape[1]
    x0_grid_trimmed = x0_grid.reshape(grid_size,1)
    x1_grid_trimmed = x1_grid.reshape(grid_size,1)
    X_simulated = np.concatenate((x0_grid_trimmed, x1_grid_trimmed), axis=1)
    
    ### Make X_simulated random indexed
    #np.random.shuffle(X_simulated)
    
    ### Transform original data distribution into Gaussian distribution
    if (flag_normlize==True):
        X_simulated = scaler.transform(X_simulated)
    else:
        ### Do not apply normalization ###
        pass
    ### Extende the 2D X_simulated to 3D
    
    if (flag_add_full_one == True):
        X_simulated_new = np.concatenate( (X_simulated, np.ones(grid_size).reshape(grid_size,1)), axis=1)
    else:
        X_simulated_new = copy.deepcopy(X_simulated)
    
    #model.predict()
    Y_sim_pred = model.predict_classes(X_simulated_new)
    
    ### Re-calculate the grid 
    if (flag_normlize==True):
        X_grid_inverse = scaler.inverse_transform(X_simulated[:,0:2])
    else:
        ### Do not apply inverse normalization ###
        X_grid_inverse = copy.deepcopy(X_simulated)
    
    x0_grid = X_grid_inverse[:,0].reshape(x0_grid_shape)
    x1_grid = X_grid_inverse[:,1].reshape(x0_grid_shape)
    Y_sim_grid = Y_sim_pred.reshape(x0_grid_shape)
    
    ### Plot the contour and disrete data points 
    color_sim = ['grey' if i==1 else 'white' for i in Y_sim_pred ]
    return x0_grid, x1_grid, X_grid_inverse, Y_sim_grid, color_sim


df_simple = pd.read_csv(fname_simple, sep=' ', header=None)
df_complex = pd.read_csv(fname_complex, sep=' ', header=None)

'''
What does each column mean:
pO2 - pCrO3 - Yes - Not
'''

df_simple.rename(columns={0:"pO2", 1:"pCro2", 2:"poisonedYes", 3: "poisonedNo"}, inplace= True)


### Visualize the poisoning data
shape = df_simple.shape

#plt.figure()
#for i in range(shape[0]):
#    if df_simple.values[i,2] == 0:
#        plt.scatter(df_simple.values[i,0], df_simple.values[i,1], color='g', marker='o')
#    else:
#        plt.scatter(df_simple.values[i,0], df_simple.values[i,1], color='r', marker='*')
##plt.scatter(df_simple.values[:,0], df_simple.values[:,1], c=df_simple.values[:,2])
#plt.xlabel('pO2')
#plt.ylabel('pCrO3')
#plt.title('Poisoning data visualization')
#plt.savefig('Original_data.pdf', format='pdf')
#plt.show()

### Normalize data ### 
X_orig = df_simple.values[:,0:2]
Y_orig = df_simple.values[:,2] # 1D output vector for Neural Networks

scaler = StandardScaler()
scaler.fit(X_orig)
'''
Apply transformation from initial dist to normal Gaussian distribution
'''
if flag_normlize == True:
    X = scaler.transform(X_orig)
else:
    '''
    Just copy original data
    '''
    X = copy.deepcopy(X_orig)

Y = copy.deepcopy(Y_orig)

X_full_zero = np.ones(shape[0]).reshape(shape[0], 1)

### If you want to add the full one column or not
if( flag_add_full_one == True):
    X = np.concatenate((X, X_full_zero), axis=1)
else:
    pass

#plt.figure()
#plt.scatter(X[:,0], X[:,1], c=df_simple.values[:,2])
#plt.xlabel('pO2')
#plt.ylabel('pCrO3')
#plt.title('Poisoning data visualization (normalized)')
#plt.savefig('Normalized_data.pdf', format='pdf')
#plt.show()


### Split whole data into training and testing dataset ###
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)

### NEURAL NETWORK CLASSIFIER ###
dim_input = X.shape[1]

model = getANN(dim_input, activ_func) 
### Save weights to re-initialize the weights ###
model.save_weights('model.h5')


if not os.path.exists(contour_dir):
    os.makedirs(contour_dir)

if flag_plot_tmp_contour == True:
    plt.figure()
#    plt.axis('equal')

hist_dict = {}
gene_acc = []
for iter_test in range(num_test):    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= ratio_test)
    
    st_time = time.time()
    ### Train the model only epochs_step once and then make contour plot ###
    
    ### If want to plot contour then set epochs small (epochs_step), else large (epochs).
    if flag_plot_tmp_contour == True:
        contour_dir_each_test = contour_dir + "/test_num_" + str(iter_test)
        if not os.path.exists(contour_dir_each_test):
            os.makedirs(contour_dir_each_test)
        
        ### Plot the initial contour (without any training)### 
        curr_epochs = 0
        x0_grid, x1_grid, X_grid_inverse, Y_sim_grid, color_sim = getLabelColorInGrid(X_orig, scaler, model, flag_normlize, flag_add_full_one)
        pred_acc = model.evaluate(X, Y, verbose=0)
        ### Make contour plotting
        plt.contour(x0_grid, x1_grid, Y_sim_grid, colors = 'blue', linewidths=1)
#            plt.hold(True) # Hold on
        plt.scatter(X_grid_inverse[:,0], X_grid_inverse[:,1], c= color_sim, s= 6, marker='s')
        
        for i in range(shape[0]):
            if df_simple.values[i,2] == 0:
                plt.scatter(df_simple.values[i,0], df_simple.values[i,1], color='g', marker='o', s= 60)
            else:
                plt.scatter(df_simple.values[i,0], df_simple.values[i,1], color='r', marker='*',s = 60)
        plt.title("Contour at epoch step " + str(curr_epochs) + " (prediction accuracy: "+ str(round(pred_acc[1],2)) + ")")
#        plt.axes().set_aspect('equal')
        plt.savefig(contour_dir_each_test+'/curr_epochs_'+str(curr_epochs).zfill(3)+'.png', format='png')
        plt.clf()
        
        curr_epochs += epochs_step
        
        while (curr_epochs <= epochs):
            print("Current epochs-> "+ str(curr_epochs))
            history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs_step, verbose=0, validation_data=(X_test, Y_test))
            x0_grid, x1_grid, X_grid_inverse, Y_sim_grid, color_sim = getLabelColorInGrid(X_orig, scaler, model, flag_normlize, flag_add_full_one)
            pred_acc = model.evaluate(X_test, Y_test, verbose=0)
            ### Make contour plotting
            plt.contour(x0_grid, x1_grid, Y_sim_grid, colors = 'blue', linewidths=1)
#            plt.hold(True) # Hold on
            plt.scatter(X_grid_inverse[:,0], X_grid_inverse[:,1], c= color_sim, s= 6, marker='s')
            for i in range(shape[0]):
                if df_simple.values[i,2] == 0:
                    plt.scatter(df_simple.values[i,0], df_simple.values[i,1], color='g', marker='o',  s= 60)
                else:
                    plt.scatter(df_simple.values[i,0], df_simple.values[i,1], color='r', marker='*',  s= 60)
            
            plt.title("Contour at epoch step " + str(curr_epochs) + " (prediction accuracy: "+ str(round(pred_acc[1],2)) + ")")
#            plt.axes().set_aspect('equal')
            plt.savefig(contour_dir_each_test+'/curr_epochs_'+str(curr_epochs).zfill(3)+'.png', format='png')
#            plt.hold(False)
            plt.clf()
            
            curr_epochs += epochs_step
    else:
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_test, Y_test))
    
    ### Save history instance dictionary ###
    hist_dict[iter_test] = history
    
    end_time = time.time()
    ### Get the generalization accuracy ###
    gene_score = model.evaluate( X_test, Y_test, verbose=1)
    gene_acc.append(gene_score[1])
    print("\nTest No.-> " + str(iter_test) + " Generalization accuracy->**" + str(round(gene_score[1], 2)) )
    print("Test No.-> " + str(iter_test) +", epochs->" + str(epochs) + ", timecost->**" + str(round(end_time-st_time, 3)))
    if( gene_score[1]>0.95):
        model.save_weights('model_good.h5')
    ### Re-initialize the weights ###
    if (iter_test < num_test-1):
#        model.load_weights('model.h5')
#        model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
        reset_weights(model)

### If there is only one hidden layer, then get the weights
if (num_test==1):
    layers = model.layers
    param_layer = {}
    param_layer[0] = layers[0].get_weights()
    param_layer[1] = layers[1].get_weights()
    
    weight_layer_0 = param_layer[0][0]
    bias_layer_0   = param_layer[0][1]
    weight_layer_1 = param_layer[1][0]
    bias_layer_1   = param_layer[1][1]
    np.savetxt('weight_layer_0.txt', weight_layer_0, delimiter=',')
    np.savetxt('bias_layer_0.txt', bias_layer_0, delimiter=',')
    np.savetxt('weight_layer_1.txt', weight_layer_1, delimiter=',')
    np.savetxt('bias_layer_1.txt', bias_layer_1, delimiter=',')
    
    weight_layer_2 = layers[2].get_weights()[0]
    bias_layer_2 = layers[2].get_weights()[1]
    weight_layer_3 = layers[3].get_weights()[0]
    bias_layer_3 = layers[3].get_weights()[1]
    
    np.savetxt('weight_layer_2.txt', weight_layer_2, delimiter=',')
    np.savetxt('bias_layer_2.txt', bias_layer_2, delimiter=',')
    np.savetxt('weight_layer_3.txt', weight_layer_3, delimiter=',')
    np.savetxt('bias_layer_3.txt', bias_layer_3, delimiter=',')



gene_acc_aver = sum(gene_acc) / float(len(gene_acc))

print("Average generalization accuracy-> " + str(gene_acc_aver))


'''
SIMULATE DATA AND APPLY NEURAL NETWORK AGAIN
'''

### Get grid, labels and colors of the discrete grid ###
x0_grid, x1_grid, X_grid_inverse, Y_sim_grid, color_sim = getLabelColorInGrid(X_orig, scaler, model, flag_normlize, flag_add_full_one)

plt.figure()
plt.contour(x0_grid, x1_grid, Y_sim_grid, colors = 'blue', linewidths=1)
plt.scatter(X_grid_inverse[:,0], X_grid_inverse[:,1], c= color_sim, s= 6, marker='s')
for i in range(shape[0]):
    if df_simple.values[i,2] == 0:
        plt.scatter(df_simple.values[i,0], df_simple.values[i,1], color='g', marker='o',  s= 60)
    else:
        plt.scatter(df_simple.values[i,0], df_simple.values[i,1], color='r', marker='*',  s= 60)
plt.title("Final contour on the grid (" + str( round(gene_acc[-1], 2)) +")")
plt.savefig(contour_dir+"/final_simulated_contour.pdf", format='pdf')
plt.show()

acc_loss_dir = "accuracy_loss"
if flag_plot_tmp_contour == False:
    if not os.path.exists(acc_loss_dir):
        os.makedirs(acc_loss_dir)
    
    ### Get sum of  accuracy and loss ###
    for i in range(len(hist_dict)):
        history = hist_dict[i]
        
        if (i==0):
            acc_train_aver = np.zeros(epochs)
            acc_valid_aver = np.zeros(epochs)
            loss_train_aver = np.zeros(epochs)
            loss_valid_aver = np.zeros(epochs)
        
        acc_train_aver  += np.array(history.history['acc'])
        acc_valid_aver  += np.array(history.history['val_acc'])
        loss_train_aver += np.array(history.history['loss'])
        loss_valid_aver += np.array(history.history['val_loss'])
    
    ### Get average of accuracy and loss ###
    acc_train_aver /= float(num_test)
    acc_valid_aver /= float(num_test)
    loss_train_aver /= float(num_test)
    loss_valid_aver /= float(num_test)
    
    
    epochs_array = np.arange(epochs)
    
    plt.figure()
    plt.plot(epochs_array, acc_train_aver)
    plt.plot(epochs_array, acc_valid_aver)
    plt.title('model accuracy (average) ')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim([0.4,1.1])
    plt.grid()
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(acc_loss_dir+"/accuracy_aver.pdf", format='pdf')
    
    
    plt.figure()
    plt.plot(epochs_array, loss_train_aver)
    plt.plot(epochs_array, loss_valid_aver)
    plt.title('model loss (average) ')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(acc_loss_dir+"/loss_aver.pdf", format='pdf')








