#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:58:00 2018

@author: charlie
"""

import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
import time


'''
call cursor.execute() for all rows **timecost -> 50s
'''
#Reading files timecost-> 0.024342775344848633
#Building connection timecost-> 0.001949310302734375
#Executing query timecost-> 49.57159662246704
#Closing connection timecost-> 0.0029098987579345703

'''
Call cursor.executemany() with list of tuple as input -> 0.7s

#    Executing query using *executemany* timecost-> 0.7178053855895996
'''

'''
Calling dataframe.to_sql() **timecost -> 1.2s**

#sqlalchemy solution timecost-> 1.176743984222412
'''

def text2sql(mydb, cursor):
    demandPath = 'demand_hours_in_year.txt'
    renewPath = 'renewables_shares.txt'
    
    st = time.time()
    demand = pd.read_csv(demandPath, sep='\t')
    demandColNames = demand.columns.values.tolist()
    demand.rename(index=str, columns={demandColNames[0]: "timestamps", 'load':'loads'}, inplace = True)
    demandColNames = demand.columns.values.tolist()
    demand.timestamps.replace('t', '', regex =True, inplace=True)
    
    renew = pd.read_csv(renewPath, sep='\t')
    renewColNames = renew.columns.values.tolist()
    renew.rename(index=str, columns={renewColNames[0]: "timestamps"}, inplace=True)
    renewColNames = renew.columns.values.tolist()
    renew.timestamps.replace('t', '', regex =True, inplace=True)
    
    end = time.time()
    print("\n Reading files timecost-> " + str(end-st))
    
    st = time.time()
    
    end = time.time()
    print("\n Building connection timecost-> " + str(end-st))
    ## CREATE TABLE
    #createTable = "create table load (timestamp int not null primary key, demand int)"
    
    '''
        ** Cursor.execute() **
    '''
    st = time.time()
    insertLoads= "INSERT INTO loads (timestamps, loads) VALUES (%s, %s)"
    insertPowerShare = "INSERT INTO powerShare( timestamps, pv, wind) VALUES (%s, %s, %s)"
    
    cursor.execute("CREATE TABLE IF NOT EXISTS loads (timestamps INT NOT NULL PRIMARY KEY, loads FLOAT(10,5) NOT NULL )")
    cursor.execute("CREATE TABLE IF NOT EXISTS powerShare ( timestamps INT NOT NULL, pv FLOAT(10,5) NOT NULL, wind FLOAT(10,5) NOT NULL, PRIMARY KEY(timestamps) )")
    mydb.commit()
    
    demand_shape = demand.shape
    
    for i in range(demand_shape[0]):
        if( demand.values[i,0] == renew.values[i,0] ):
            cursor.execute(insertLoads, ( demand.values[i,0], demand.values[i,1]))
            cursor.execute(insertPowerShare, (renew.values[i,0], renew.values[i,1], renew.values[i,2]) )
    mydb.commit()
    
    end = time.time()
    print("\n Executing query using *execute* timecost-> " + str(end-st))
    
    
    st =time.time()
    mydb.commit()
    
    end = time.time()
    print("\n Closing connection timecost-> " + str(end-st))
    
    
    '''
        ** Cursor.executemany() **
    '''
    
    st = time.time()
    insertLoads= "INSERT INTO loads_many (timestamps, loads) VALUES (%s, %s)"
    insertPowerShare = "INSERT INTO powerShare_many( timestamps, pv, wind) VALUES (%s, %s, %s)"
    
    cursor.execute("CREATE TABLE IF NOT EXISTS loads_many (timestamps INT NOT NULL PRIMARY KEY, loads FLOAT(10,5) NOT NULL )")
    cursor.execute("CREATE TABLE IF NOT EXISTS powerShare_many ( timestamps INT NOT NULL, pv FLOAT(10,5) NOT NULL, wind FLOAT(10,5) NOT NULL, PRIMARY KEY(timestamps) )")
    mydb.commit()
    end = time.time()
    print('\n Creating tables *loads_many* and powerShare_many timecost -> ' + str(end-st))
    
    st = time.time()
    demand_list = [(item[0], item[1]) for item in demand.values]
    renew_list = [(item[0], item[1], item[2]) for item in renew.values]
    
    cursor.executemany(insertLoads, demand_list)
    cursor.executemany(insertPowerShare, renew_list )
    #mydb.commit()
    
    end = time.time()
    print("\n Executing query using *executemany* timecost-> " + str(end-st))
    
    
    
    '''
        ** dataframe.to_sql() **
    '''
    st = time.time()
    cursor.execute("CREATE TABLE IF NOT EXISTS loads_alchemy (timestamps INT NOT NULL PRIMARY KEY, loads FLOAT(10,5) NOT NULL )")
    cursor.execute("CREATE TABLE IF NOT EXISTS powerShare_alchemy ( timestamps INT NOT NULL, pv FLOAT(10,5) NOT NULL, wind FLOAT(10,5) NOT NULL, PRIMARY KEY(timestamps) )")
    mydb.commit()
    end = time.time()
    print('\n Creating *loads_alchemy* and powerShare_alchemy timecost -> ' + str(end-st))
    st = time.time()
    ### Approach two-> USING SQLALCHEMY
    engine = create_engine('mysql+mysqlconnector://root:charlieli@localhost/eonTest', echo=False)
    demand.to_sql('loads_alchemy', con=engine, if_exists='replace', index=False)
    renew.to_sql('powerShare_alchemy', con=engine, if_exists = 'replace', index=False)
    end = time.time()
    print('\n sqlalchemy solution timecost-> ' + str(end-st))


def _readColumnFromDB(cursor, tableName, colIdx = 1, rowStart=0, numEntries=10):
    '''
    read a column from a database
    ### TODO: input parameter-> sqlLine
    ##* limit rowStart, numEntry *## rowStart-> index from 0
    '''
    
    ### Get the column names of the table 'tableName'
    cursor.execute('SELECT * FROM ' + tableName + ' LIMIT 0')
    tableColName = cursor.column_names
    cursor.fetchall() # Required to execute again, or -> unread result found
    
    sqlStr = "SELECT " + tableColName[colIdx] + " From " + tableName + " LIMIT "\
            + str(rowStart) + ", " + str(numEntries)
#    sqlStr = "SELECT loads FROM loads LIMIT " + str(rowStart) + ", " + str(numEntries)
    colData = None
    try:
        cursor.execute(sqlStr)
        colData = cursor.fetchall() ## colData is a list of tuple 
    except Exception as e:
            print(":( Something wrong when reading column from database")
            print(e)
    ### Convert list of tuple to float list
    result = [item[0] for item in colData]
    return result


if __name__ == '__main__' :
    ## CREATE CONNECTION 
    mydb = mysql.connector.connect(
            host = "localhost",
            user = "root",
            passwd = "youknow",
            database = "eonTest"
            )
    
    cursor = mydb.cursor()
    pass
    try:
        pass
        resData = _readColumnFromDB(cursor, tableName="loads", colIdx=1, rowStart=0, numEntries=10)
        print("Successfully got the result")
        print(resData)
#        text2sql(mydb, cursor)
    except Exception as e:
        print(e)
        cursor.close()
        mydb.close()
    finally:
        cursor.close()
        mydb.close()
