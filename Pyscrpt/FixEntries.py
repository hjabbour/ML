#!/usr/bin/env python
# coding: utf-8

# ## Develop this to have entrylist with highest entry lowest and average

# ## maybe entry from the average trade script

# In[13]:


import numpy as np 
import pandas as pd 
from datetime import date
#Data Source (Pandas Data Reader)
from pandas_datareader import data
import requests 
import json
from dateutil.relativedelta import relativedelta


#Data Source(Pandas Data Reader)


# In[14]:


#entrylist =[['ANKR/USDT',0.07684],['AVAX/USDT',57],['WAXP/USDT',0.3818],['REEF/USDT',0.01346],['LINK/USDT',14.15],['AR/USDT',30],['ATOM/USDT',23],['MIR/USDT',2.343],['ALGO/USDT',0.909],['ZRX/USDT',0.90],['XTZ/USDT',2.856],['KSM/USDT',163],]
#entrylist =[['ANKRUSDT',0.07684],['AVAXUSDT',57],['WAXPUSDT',0.3818],['REEFUSDT',0.01131],['LINKUSDT',14.15],['ARUSDT',30],['ATOMUSDT',23],['MIRUSDT',2.343],['ALGOUSDT',0.909],['ZRXUSDT',0.60],['XTZUSDT',2.856],['KSMUSDT',163],['NEARUSDT',14],['WINUSDT',0.0003138],['CELRUSDT',0.05063],['DENTUSDT',0.002459],['LRCUSDT',0.7047],['CRVUSDT',3.529],['IMXUSDT',2.709],['SCRTUSDT',5],['RENUSDT',0.2547],['SUSHIUSDT',4.522],['CAKEUSDT',11.24],['XLMUSDT',0.2109],['HNTUSDT',30],['OMGUSDT',4.5],['LINAUSDT',0.024],['MIRUSDT',2.343],['FLUXUSDT',1.523]]
entrylist =[['ANKRUSDT',0.05415],['AVAXUSDT',57],['WAXPUSDT',0.3174],['REEFUSDT',0.00930],['LINKUSDT',14.15],['ARUSDT',30],['ATOMUSDT',23],['MIRUSDT',2.343],['ALGOUSDT',0.909],['ZRXUSDT',0.50],['XTZUSDT',2.856],['KSMUSDT',150],['NEARUSDT',10.320],['WINUSDT',0.0003138],['DENTUSDT',0.002459],['LRCUSDT',0.677],['CRVUSDT',2.12],['IMXUSDT',1.30],['SCRTUSDT',5],['RENUSDT',0.3047],['SUSHIUSDT',3.5],['CAKEUSDT',11.24],['XLMUSDT',0.2109],['HNTUSDT',30],['OMGUSDT',4.5],['LINAUSDT',0.01759],['FLUXUSDT',1.523],['SRMUSDT',1.8],['VTHOUSDT',0.0031],['ANTUSDT',5.616],['RAYUSDT',2.5],['ROSEUSDT',0.24055],['MOVRUSDT',51.6],['TFUELUSDT',0.22],['GLMRUSDT',4.5],['COTIUSDT',0.2277],['ONEUSDT',0.13171],['VETUSDT',0.05138],['RUNEUSDT',3.709],['CELRUSDT',0.034063],['ONTUSDT',0.4652],['GALAUSDT',0.24794],['GRTUSDT',0.3164],['DOTUSDT',16.31],['ZILUSDT',0.04410],['FILUSDT',20],['SCUSDT',0.015],['MATICUSDT',1.4],['FTMUSDT',1.0],['BANDUSDT',3.6],['1INCHUSDT',1.6],['AUDIOUSDT',1.0]]
exposurelist =[['ANKRUSDT',311],['AVAXUSDT',2850],['WAXPUSDT',527],['REEFUSDT',603],['LINKUSDT',6000],['ARUSDT',2450],['ATOMUSDT',1863],['MIRUSDT',85],['ALGOUSDT',2735],['ZRXUSDT',245],['XTZUSDT',1200],['KSMUSDT',7800],['NEARUSDT',1360],['WINUSDT',285],['DENTUSDT',385],['LRCUSDT',660],['CRVUSDT',323],['IMXUSDT',269],['SCRTUSDT',601],['RENUSDT',684],['SUSHIUSDT',2730],['CAKEUSDT',4000],['XLMUSDT',368],['HNTUSDT',999],['OMGUSDT',765],['LINAUSDT',205],['FLUXUSDT',224],['SRMUSDT',587],['VTHOUSDT',413],['ANTUSDT',200],['RAYUSDT',100],['ROSEUSDT',634],['MOVRUSDT',500],['TFUELUSDT',244],['GLMRUSDT',400],['COTIUSDT',500],['ONEUSDT',1000],['VETUSDT',500],['RUNEUSDT',750],['CELRUSDT',500],['ONTUSDT',400],['GALAUSDT',650],['GRTUSDT',1700],['DOTUSDT',10000],['ZILUSDT',1000],['FILUSDT',2000],['SCUSDT',200],['MATICUSDT',8000],['FTMUSDT',2000],['BANDUSDT',750],['1INCHUSDT',1000],['AUDIOUSDT',700]]
startdate = '2021-01-01'
enddate = '2022-02-22'
enddate = pd.to_datetime('today')

interval = '1h'
easyband = pd.read_csv('easyband.csv')
#Columns=['Indicator','ALLTIME','CLOSERATIO','SMA100','SMA55']
easyband = pd.DataFrame(easyband,columns=['Indicator','ALLTIME','CLOSERATIO','SMA21','SMA55','GREENEASY'])
easyband = easyband.loc[:,['Indicator','SMA21','GREENEASY']]

#EASY=float(easyband.loc[(easyband.Indicator=='BANDUSDT')].SMA100.values)
#print(easyband)
#EASY[0]
#easyband


# In[15]:


entries = pd.read_csv('entries.csv')

entries = pd.DataFrame(entries, columns =['Indicator', 'lowest_entry']) 

#entries = pd.DataFrame(entrylist, columns =['Indicator', 'lowest_entry'],dtype = float) 
entries
entries.lowest_entry.astype(float)
indicator_list = entries['Indicator']
#indicator_list
#entries
entries.to_csv("entries.csv")

exposure = pd.DataFrame(exposurelist, columns =['Indicator', 'exposure'],dtype = float) 
exposure
exposure.exposure.astype(float)
#indicator_list
#entries
entries.to_csv("exposure.csv")

entries = entries.merge(exposure ,how="left",on="Indicator")
#entries


# In[16]:


#whitelist=whitelist.lstrip()
#whitelist = indicator_list
#array = whitelist.split(",")
array = indicator_list
wl = []
bl = []
blacklist = ""
whitelist = ""
symbols =[]
for i in range(0 ,len(array)):
    if (array[i].find('DOWN') != -1):
        bl.append(str(array[i]).lstrip())
        #print ("Contains given substring ")
        blacklist += str(array[i])
    elif (array[i].find('UP') != -1):
        bl.append(str(array[i]).lstrip())
        #print ("Contains given substring ")
        blacklist += str(array[i])

    else:
        wl.append(str(array[i]).lstrip().replace("/",""))
        whitelist +=str(array[i]).lstrip()
        #print ("Doesn't contains given substring")
    

    
#symbols to be used to get the prices from binance url
symbols = np.array(wl)


## get the data 
import os

live = pd.to_datetime('today')
start = pd.to_datetime(startdate)
end = pd.to_datetime(enddate)

root_url = 'https://api.binance.com/api/v1/klines'
symbol = 'STEEMETH'
interval = '1d'
url = root_url + '?symbol=' + symbol + '&interval=' + interval
print(url)


directory = 'BN-main'
table = []
   
        
        

for symbol in symbols:
        url = root_url + '?symbol=' + symbol + '&interval=' + interval
        data = json.loads(requests.get(url).text)
        dflive = pd.DataFrame(data ,columns =['open_time',
              'o', 'h', 'l', 'c', 'v',
              'close_time', 'qav', 'num_trades',
              'taker_base_vol', 'taker_quote_vol', 'ignore'])
        dflive.open_time = pd.to_datetime(dflive['open_time'],unit='ms')
        dflive=dflive.loc[(dflive.open_time>start)]
        lastrow = dflive.iloc[-1]
        now= float(lastrow.c)
        alth = dflive.h.max()
        altl = dflive.l.min()
        
    
        ## lowest entiry
        lowest = float(entries.loc[(entries.Indicator==symbol)].lowest_entry.values)
        #print(lowest)
        easyseries = easyband.loc[(easyband.Indicator==symbol)].GREENEASY.values
        #print(easyseries)
        if easyseries.size>0:
                green = easyseries[0]
        else:
                green = 0
        oeasyseries = easyband.loc[(easyband.Indicator==symbol)].SMA21.values
        #print(easyseries)
        if oeasyseries.size>0:
                orange = oeasyseries[0]
        else:
                orange = 0
 
        tuple =(symbol,alth,now,lowest,green,orange)
        table.append(tuple)
        #print("name is: ",symbol ,alth,altl)
        

table[0]
AllTime = pd.DataFrame(table,columns=['Indicator','ATH','Current','LOWEST','GREEN','ORANGE'])
AllTime.ATH=AllTime.ATH.astype(float)
#AllTime.ATL=AllTime.ATL.astype(float)
AllTime.Current=AllTime.Current.astype(float)
AllTime.LOWEST=AllTime.LOWEST.astype(float)
AllTime.GREEN=AllTime.GREEN.astype(float)
AllTime.ORANGE=AllTime.ORANGE.astype(float)




AllTime.dtypes

#AllTime['ALLTIME']=(AllTime['Current']/AllTime['ATH'])*100
#AllTime['MONTH']=(AllTime['Current']/AllTime['OMH'])*100
#AllTime['2MONTH']=(AllTime['Current']/AllTime['TMH'])*100
#AllTime['3MONTH']=(AllTime['Current']/AllTime['THMH'])*100
AllTime['DROP']=((AllTime['Current']-AllTime['LOWEST'])/AllTime['LOWEST'])*100

#AllTime.sort_values(by=['ALLTIME'],ascending=True,inplace = True)
#AllTime.sort_values(by='ATH',ascending=False)
#AllTime.to_csv("alltimeHL.csv", mode="w")
#data.sort_values("Name", axis = 0, ascending = True,inplace = True, na_position ='last')

## Display the Data 
AllTime.shape


# In[17]:


## Display the Data coins run last  Month

#drop = AllTime.loc[(AllTime['MONTH'] < 35)]
#drop = AllTime.loc[(AllTime['MONTH'] < 70)]
#drop = AllTime.loc[(AllTime['MONTH'] < 80)&(AllTime['2MONTH'] < 70)&(AllTime['ALLTIME'] < 40)]
drop = AllTime.loc[(AllTime['Current'] <= AllTime['LOWEST'])]

#sort = drop.sort_values(by=['EASY'],ascending=False,inplace = True)
sort = drop.sort_values(by=['DROP'],ascending=True,inplace = True)

#drop = drop.loc[(drop['ALLTIME'] < drop['3MONTH'])]
display = drop[['Indicator','ATH','ORANGE','GREEN','DROP','LOWEST','Current']]

#print(display.iloc[0:60,0:14])
#print(display.iloc[60:110,])
#print(display.iloc[110:160,])
print("\b Fix Lowest entry ")

display = display.merge(exposure ,how="left",on="Indicator")
#display1 = display.sort_values(by=['exposure'],ascending=True,inplace = True)


display.to_csv("fixentries.csv")

display


# In[18]:


display = display.loc[:]
display = display.loc[display.exposure > 100]
display


# In[19]:


## Green entry
print("\b Enries on easy band")

display1 = display.loc[:]
display1


# In[20]:


display1['Value']=(display['DROP']/100)*display['exposure']
display1


# In[21]:


display = display1.loc[:]
#display['RED']=display['ORANGE']*2
display['Value']=(display['DROP']/100)*display['exposure']
sort = display.sort_values(by=['Value'],ascending=True,inplace = True)

#display['Ratio']=display['ATH']/display['Current']
#display['ORatio']=display['ORANGE']/display['Current']
#display = display.loc[display['Current'] <= display['GREEN']]
display


# In[22]:


#del display ,sort, AllTime


# In[23]:


#AllTime


# In[ ]:




