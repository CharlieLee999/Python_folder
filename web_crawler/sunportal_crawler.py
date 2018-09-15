# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 14:12:10 2018

@author: charlie
"""

from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import TimeoutException, WebDriverException
import urllib.request
import time


#############################################################

#        How many days do you want? (From today)
#        total_days: integer 
#############################################################

total_days = 500
counter = 0

browser = webdriver.Chrome()
browser.get("https://www.sunnyportal.com/StatusMonitor")

current_url = browser.current_url
source = urllib.request.urlopen(current_url).read()

#time.sleep(5)
# ADD a breakpoint here
browser.switch_to.window(browser.window_handles[-1])

current_url = browser.current_url
source = urllib.request.urlopen(current_url).read()

def downloadFiles(total_days, counter):
    for i in range(total_days):
        browser.execute_script("document.getElementsByClassName('PC_pagebtn')[0].click()")
        time.sleep(2.5)  
#        browser.execute_script("document.getElementsByClassName('PC_DownloadButton')[0].click()")
#        time.sleep(1.5)
        counter = counter+1
    return counter

while(counter<total_days):
    try:
        counter = downloadFiles(2, counter)
        print("Current counter is " + str(counter)+" NO ERROR")
    except WebDriverException as error: 
        print(error)
        print("Current counter is " + str(counter))
