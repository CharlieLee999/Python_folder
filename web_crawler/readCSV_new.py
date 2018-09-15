#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 08:46:58 2018

DEBUG FILE:
    
#csvPath = '/home/charlie/Downloads/Energiebilanz/Energiebilanz_2017_09_16.csv'
    csvPath = '/home/charlie/Downloads/Energiebilanz/Energiebilanz_2018_08_31.csv'
@author: charlie
"""

import numpy as np
import pandas as pd
import glob 
import math
import sys

inPath = "/home/charlie/Downloads/Energiebilanz/"
outPath = "output.csv"
inputPaths = glob.glob(inPath+"*.csv")
inputPaths.sort()



# day: inputPaths[i][-6:-4]
# Month: inputPaths[i][-9:-7]
# Year: inputPaths[i][-14:-10]

'''
    original csvFile data instructions:
        time_tag:           csvFile.values[:,0] 
        direktbrauch:       csvFile.values[:,1]
        Batterieentladung:  csvFile.values[:,2]
        Netzbezug:          csvFile.values[:,3]
        Gesamtverbrauch:    csvFile.values[:,4]
        Netzeinspeisung:    csvFile.values[:,5]
        Direktverbrauch:    csvFile.values[:,6]
        Batterieladung:     csvFile.values[:,7]
        PV-Erzeugung:       csvFile.values[:,8]
        Begrenzung der Wirkleistungseinspeisung: csvFile.values[:,9]
        
        
    new output.csv data instructions:
        same as the original one apart from inserting three year, month and day columns before time_tag (hh_mm)
'''

## IF curValue is string:
#        subcases:
#            1: "1,22"-> 1,22kw -> 1220
#            2: "1.22"-> 1.22kw -> 1220
#            3: "122" -> 122w   -> 122
# IF curValue is numeric value
# Check it is 1.222 or 122
def convert2Float(curValue):
    ret = curValue
    if( type(curValue) == str):
        if("," in curValue):
            ret = float(curValue.replace(',', '.'))*1000
        else:
            ret = float(curValue)            
            if( (ret-math.floor(ret))> 0.001):
                ret = ret * 1000
            else:
                pass
        
    elif( (curValue- math.floor(curValue))>0.001 ): # Float
        ret = curValue*1000
    return ret

#csvFile = pd.read_csv(inputPaths[0], sep=';')
#csvFile2 = pd.read_csv(inPath+"Energiebilanz_2018_08_20.csv", sep=";")
#csvFile3 = pd.read_csv(inPath+"Energiebilanz_2018_08_25.csv", sep=";")


# Germany currency to US current(float)
# newValue = float(old.replace(',', '.'))
# if (type(b) == str)
# b= csvFile3.values[28,-1]


# 53, 120 and other value smaller than 1000 are interger now.
# Want to transform these wrong code value to float value like 0.053, 0.12
# a = csvFile.values[28,-1]
# if ( a-math.floor(a) == 0.0) then a should be wrong data 

totalList = []
counter = 0
for csvPath in inputPaths: #[300:-1]
    
    # Get year-month-day
    year = int(csvPath[-14:-10])
    month = int(csvPath[-9:-7])
    day = int(csvPath[-6:-4])
    
    # Read the csv
    csvFile = pd.read_csv(csvPath, sep=';')
    pvList = csvFile.values[:, -1]
    headers = csvFile.columns.values.tolist()
    
    # Get number of rows
    numRow = csvFile.shape[0]
    numCol = csvFile.shape[1]
    
    #  Convert and store all float values
    # Convert the irregular data to regular data
                # e.g. 0.1kw -> 100.0w
    
    for colIdx in range(1, numCol):
        for rowIdx in range(numRow):
            curValue = csvFile.values[rowIdx, colIdx]
#            print("curValue is " + str(curValue) + ", type is " + str(type(curValue)) + ", rowIdx, colIdx are " + str(rowIdx) + ", " + str(colIdx) )
            tmp = curValue
            
            # IF curValue is whitespace, then set it as nan
            if(curValue==' '):
                curValue = math.nan
                print('curValue is whitespace, ROW and Col idx are  ' + str(rowIdx) + ', '+str(colIdx))
            # If current data is NAN, then apply the two point interpolation as the average between the previous and next value
            
            if((type(curValue) is not str) and math.isnan(curValue)):
                preValue = csvFile.values[(rowIdx-1)%numRow, colIdx]
                nextValue = csvFile.values[(rowIdx+1)%numRow, colIdx]
                
                # Set the whitespace as nan
                if( preValue == ' '):
                    print('preValue is whitespace, ROW and Col idx are  ' + str(rowIdx-1) + ', '+str(colIdx))
                    preValue = math.nan
                if( nextValue == ' '):
                    nextValue = math.nan
                    print('nextValue is whitespace, ROW and Col idx are  ' + str(rowIdx+1) + ', '+str(colIdx))
                # IF curValue is the first row, then rowIdx-1 = -1 = last row
                
#                if( rowIdx==0 ):
#                    preValue = 0
                # IF the preValue is also nan, then set preValue as 0
                if( (rowIdx==0) and ((type(preValue) is not str) and math.isnan(preValue)) ):
                    preValue = 0
                else:
                    preValue = convert2Float(preValue)
                    
                # If the nextValue is also nan, then set curValue as modified preValue.
                if((type(nextValue) is not str) and math.isnan(nextValue)):
                    csvFile.at[rowIdx, headers[colIdx]] = preValue
                else:
                    tmpFloat = convert2Float(nextValue)
                    csvFile.at[rowIdx, headers[colIdx]] = (tmpFloat+preValue)/2.0
            else:
                tmpValue = 0
                try:
                    tmpValue = convert2Float(curValue)
                except ValueError as err:
                    print(err)
                    print('curValue is **' + str(tmp) + "** rowIdx, colIdx are " + str(rowIdx) + ", " + str(colIdx) )
                    print('File name is ' + csvPath)
                finally:
#                    if(type(curValue) == float):
#                        print("Finally branch->  curValue is float, colIdx is " + str(colIdx))
#                        print("Type of int(tmpValue) is " + str(type(tmpValue)) )
                    csvFile.at[rowIdx, headers[colIdx]] = tmpValue
    
    ## Make sure all data are integer now
    csvFile[headers[1:]] = csvFile[headers[1:]].applymap(np.int16)
    
    # ADD year, month, day columns
    csvFile.insert(loc = 0, column='day', value=day)
    csvFile.insert(loc = 0, column='month', value=month)
    csvFile.insert(loc = 0, column='year', value=year)
    
    # Rename the column name of the exact time column
    new_header_time = 'hh_mm'
    csvFile.rename(columns = {' ': new_header_time}, inplace=True)
    
    # Clean the time column
    for rowIdx in range(numRow):
        cur_value = csvFile[new_header_time][rowIdx]
        cur_value = (cur_value.replace('="', '')).replace('"', '')
        csvFile.at[rowIdx, new_header_time] =  cur_value
    
    # TODO! Split the string column time hh:mm to two seperate column
    
    # Output the current Dataframe to the csvFile
    # IF first open the file, then clean the original content and write the Dataframe with header
    # IF not, then only append the Dataframe without header
    if(counter==0):
        #open(outPath, 'w+')
        csvFile.to_csv(outPath, header=True, index=False)
    else:
        csvFile.to_csv(outPath, mode='a', header=False, index=False)
#        try:
#            print( 'Output byte size is '+ str(os.stat('outPath').st_size) )
#        except FileNotFoundError as err:
#            print(err)
#            print('File does not exist, a new file will be created')
#            open(outPath, 'w')
            
    counter = counter + 1 
    print("Index of this file is " + str(counter))
    print(csvPath)
    
    
