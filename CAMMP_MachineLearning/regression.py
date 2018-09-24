#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:59:10 2018

@author: charlie
"""

import pandas as pd
import numpy as np
'''
index c mn v si cr mo ni w page bs-exp other bs 
0     1  2 3  4  5 6  7  8  9   10      
'''

fname = "DataBainite_summarizingDataBS.csv"
data = pd.read_csv(fname, sep=',', header = None)
corr_mat = data.corr()

data_shape = data.shape

num_nonzero_column =  np.count_nonzero(data.values, axis=0)

num_zero_column = data_shape[0] - num_nonzero_column