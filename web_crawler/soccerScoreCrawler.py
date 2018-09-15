###########################################################
# 		Web crawler with selenium and BeautifulSoup		  #
# 		Operate the web with selenium					  #
# 		Parse the table element with BeautifulSoup		  #
# 		AUTHOR: XING LI 								  #
###########################################################


from selenium import webdriver
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
import time


browser = webdriver.Chrome()
browser.get("https://www.bundesliga.com/de/bundesliga/tabelle/2017-2018/")


# Select the option Spieltag 1
# <option  value="?d=1"> Spieltag 1 </option>
classSelectValue = '?d=1'
Select(browser.find_element_by_id('matchday-dropdown')).select_by_value(classSelectValue)

# Let the code sleep 2 seconds to wait for the webpage refreshing
time.sleep(2);

current_url = browser.current_url
#current_url = "https://www.bundesliga.com/de/bundesliga/tabelle/2017-2018/?d=1&t=BLITZTABELLE"
source = urllib.request.urlopen(current_url).read()
soup = BeautifulSoup(source, 'lxml')
table = soup.table

table_header = ["rank", "matches", "pts", "lg wins", "lg draws", "lg looses", "lg goals", "lg difference"];

table_rows = table.find_all('tr')

trName = table_rows[0]
th = trName.find_all("th", {"class":table_header})
th_text = [i.text for i in th]
#print(th_text)

table_list = []
print(table_header)
for tr in table_rows[1:]:
    td = tr.find_all("td", {"class":table_header})
    td_text = [i.text for i in td]
    print(td_text)
    table_list.append(td_text)

table_data = np.array(table_list)
table_shape = table_data.shape

# Array used to store the goals data
gates = np.zeros((table_shape[0],2))

# Split the gates 1:2 to two int valuea
for i in range(table_shape[0]):
    gate = table_data[i][-2].split(":");
    gates[i][0] = gate[0]
    gates[i][1] = gate[1]

# Delete the title "lg goals" and replace it with "lg goals self" and "lg goals enemy"
del table_header[-2]
table_header.insert(-1, 'lg goals self')
table_header.insert(-1, 'lg goals enemy')

# Concatenate all class names to a string seperated by ","
table_header_csv=""
for i in range(table_shape[1]+1):
    if(i!=table_shape[1]):
        table_header_csv += table_header[i]+","
    else:
        table_header_csv += table_header[i]

# Replace the data of "lg goals" with "lg goals self" and "lg goals enemy"
table_data = np.delete(table_data, -2, axis=1)
table_data = np.insert(table_data, -1, gates[:,0], axis=1)
table_data = np.insert(table_data, -1, gates[:,1], axis=1)

# Transform to float fisrt because of ValueError: invalid literal for int() with base 10: '3.0'
table_data = table_data.astype(np.float)
table_data = table_data.astype(np.int)

# Save the int numpy array to csv with header: table_header_csv
np.savetxt("soccerScores.csv", table_data, delimiter=",", comments="", header=table_header_csv)
