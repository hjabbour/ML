#!/usr/bin/env python
# coding: utf-8

# Generate the easy band in the first section based on live data 
# compare it to the entries in the second part 

# In[55]:


import numpy as np 
import pandas as pd 
from datetime import date
#Data Source (Pandas Data Reader)
from pandas_datareader import data
import requests 
import json
from dateutil.relativedelta import relativedelta


#Data Source(Pandas Data Reader)


# In[56]:


whitelist = 'POWR/USDT, MDT/USDT,BLZ/USDT, CHR/USDT, ORN/USDT, VIDT/USDT, IRIS/USDT, SHIB/USDT, MATIC/USDT,ALGO/USDT, BTC/USDT, ETH/USDT, AXS/USDT, SOL/USDT, MATIC/USDT, ADA/USDT, BNB/USDT, FIL/USDT, XRP/USDT, CVC/USDT, FTM/USDT, ICP/USDT, DOGE/USDT, XEC/USDT, IOTA/USDT, AVAX/USDT, DOT/USDT, LTC/USDT, EOS/USDT, IOST/USDT, STMX/USDT, FTT/USDT, TWT/USDT, ALICE/USDT, ATA/USDT, ETC/USDT, VET/USDT, SAND/USDT, LINK/USDT, THETA/USDT, SHIB/USDT, CHZ/USDT, LUNA/USDT, IOTX/USDT, TRX/USDT, TLM/USDT, BCH/USDT, REEF/USDT, CAKE/USDT, NEO/USDT,RVN/USDT, ATOM/USDT, SUSHI/USDT, ARDR/USDT, OMG/USDT, GRT/USDT, 1INCH/USDT, AAVE/USDT, C98/USDT, UNI/USDT, HBAR/USDT, CRV/USDT, DENT/USDT, MBOX/USDT, BTT/USDT, SRM/USDT, YFI/USDT, HOT/USDT, XLM/USDT, FIS/USDT, REQ/USDT, SXP/USDT, KSM/USDT, QNT/USDT, NEAR/USDT, COMP/USDT, QTUM/USDT, CHR/USDT, SC/USDT, XVS/USDT, ONT/USDT, COTI/USDT, NKN/USDT, RUNE/USDT, ANKR/USDT, MANA/USDT, CTXC/USDT, AR/USDT, RAY/USDT, BAKE/USDT, ARPA/USDT, XTZ/USDT, ZIL/USDT, ONG/USDT, ROSE/USDT, PNT/USDT, ALGO/USDT, MINA/USDT, WIN/USDT, AUDIO/USDT, STORJ/USDT, EPS/USDT, MDX/USDT, ZEC/USDT, ENJ/USDT, HIVE/USDT, TFUEL/USDT, WRX/USDT, SNX/USDT, SKL/USDT, LINA/USDT, DODO/USDT, UNFI/USDT, XEM/USDT, EGLD/USDT, ICX/USDT, WAVES/USDT, ONE/USDT, DASH/USDT, KAVA/USDT, SUPER/USDT, TKO/USDT, OGN/USDT, XMR/USDT, OCEAN/USDT, CELO/USDT, BAT/USDT, FUN/USDT, LRC/USDT, ALPHA/USDT, MASK/USDT, TRU/USDT, DNT/USDT, LIT/USDT, FET/USDT, GTC/USDT, CELR/USDT, DEXE/USDT, ZEN/USDT, WAXP/USDT, CTSI/USDT, HNT/USDT, YFII/USDT, BEL/USDT, ZRX/USDT, DEGO/USDT, RSR/USDT, COS/USDT, FLM/USDT, BAL/USDT, MFT/USDT, RLC/USDT, KNC/USDT, REN/USDT, INJ/USDT, FLOW/USDT, SFP/USDT, BAND/USDT, CTK/USDT, SUN/USDT, AKRO/USDT, XVG/USDT, NANO/USDT, STRAX/USDT, PUNDIX/USDT, TVK/USDT, MKR/USDT, CLV/USDT, TRB/USDT, ETHUP/USDT, KEEP/USDT, TCT/USDT, POND/USDT, LTO/USDT, ANT/USDT, TOMO/USDT, ACM/USDT, FIO/USDT, VITE/USDT, MBL/USDT, JST/USDT, MTL/USDT, UTK/USDT, BLZ/USDT, ORN/USDT, STPT/USDT, CFX/USDT, OM/USDT, PERP/USDT, MIR/USDT, STX/USDT, ERN/USDT, TUSD/USDT, PSG/USDT, NULS/USDT, OXT/USDT, REP/USDT, BNBUP/USDT, DATA/USDT, WAN/USDT, BURGER/USDT, RAMP/USDT, ATM/USDT, DGB/USDT, DIA/USDT, BEAM/USDT, UMA/USDT, ETHDOWN/USDT, CKB/USDT, TRIBE/USDT, FARM/USDT, BTCUP/USDT, BZRX/USDT, ALPACA/USDT, DOTUP/USDT, BTS/USDT, FOR/USDT, ADAUP/USDT, HARD/USDT, LSK/USDT, POLS/USDT, FORTH/USDT, TORN/USDT, VTHO/USDT, BTG/USDT, QUICK/USDT, FILUP/USDT, PAX/USDT, MITH/USDT, SUSHIUP/USDT, BTCDOWN/USDT, XRPDOWN/USDT, BNBDOWN/USDT, DUSK/USDT, XRPUP/USDT, WTC/USDT, DOCK/USDT, AUTO/USDT, BTCST/USDT, KLAY/USDT, WNXM/USDT, KEY/USDT, FILDOWN/USDT, NU/USDT, IRIS/USDT, KMD/USDT, TROY/USDT, WING/USDT, AION/USDT, PHA/USDT, SUSHIDOWN/USDT, EOSUP/USDT, LTCUP/USDT, GHST/USDT, PAXG/USDT, BNT/USDT, AVA/USDT, BADGER/USDT, PERL/USDT, COCOS/USDT, GXS/USDT, MDT/USDT, 1INCHUP/USDT, GTO/USDT, LINKUP/USDT, AAVEUP/USDT, FIRO/USDT, JUV/USDT, MLN/USDT, UNIUP/USDT, LPT/USDT, NMR/USDT, AAVEDOWN/USDT, 1INCHDOWN/USDT, ASR/USDT, OG/USDT, BAR/USDT, DREP/USDT, BOND/USDT, DCR/USDT, GNO/USDT, NBS/USDT, RIF/USDT'
#whitelist = 'LINK/USDT, DOT/USDT'
startdate = '2020-01-01'
enddate = '2022-03-22'
enddate = pd.to_datetime('today')

interval = '1w'


# In[57]:


def FRSI(series, period=14):
    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period-1]] = np.mean( ups[:period] ) #first value is sum of avg gains
    ups = ups.drop(ups.index[:(period-1)])
    downs[downs.index[period-1]] = np.mean( downs[:period] ) #first value is sum of avg losses
    downs = downs.drop(downs.index[:(period-1)])
    rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() /          downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() 
    return 100 - (100 / (1 + rs))


# calculating Stoch RSI (gives the same values as TradingView)
# https://www.tradingview.com/wiki/Stochastic_RSI_(STOCH_RSI) 
def STRSIA(series, period=14, smoothK=3, smoothD=3):
    # Calculate RSI 
    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period-1]] = np.mean( ups[:period] ) #first value is sum of avg gains
    ups = ups.drop(ups.index[:(period-1)])
    downs[downs.index[period-1]] = np.mean( downs[:period] ) #first value is sum of avg losses
    downs = downs.drop(downs.index[:(period-1)])
    rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() /          downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() 
    rsi = 100 - 100 / (1 + rs)

    # Calculate StochRSI 
    stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    stochrsi_K = stochrsi.rolling(smoothK).mean()
    stochrsi_D = stochrsi_K.rolling(smoothD).mean()
    return stochrsi, stochrsi_K, stochrsi_D

#Calulate SMA 
def SMA(series,size):
    sma = series.rolling(size).mean()
    return sma
#Calculate SMA55


# In[58]:


whitelist=whitelist.lstrip()
array = whitelist.split(",")
array
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
#interval = '1d'
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
        
        dflive = dflive.astype ({'o': float ,'h':float,'l':float ,'c':float ,'v':float})
        dflive['SMA21'] = SMA(dflive['c'],21)
        dflive['SMA55'] = SMA(dflive['c'],55)
        dflive['SMA100'] = SMA(dflive['c'],100)
        
        
        
        #print(symbol)
        lastrow = dflive.iloc[-1]
        now = float(lastrow.c)
        SMA21 = float(lastrow.SMA21)
        SMA55 = float(lastrow.SMA55)
        SMA100 = float(lastrow.SMA100)
        
        
        alth = dflive.h.max()
        altl = dflive.l.min()
        
        
        ## past 1 month 
        past_month = end - relativedelta(months=1)
        
        back1Mdata = dflive.loc[(dflive.open_time> past_month)]
        omh = back1Mdata.h.max()
        oml = back1Mdata.l.min()
        ## past 2 month
        
        past_2month = end - relativedelta(months=2)
        back2Mdata = dflive.loc[(dflive.open_time> past_2month)]
        tmh = back2Mdata.h.max()
        tml = back2Mdata.l.min()
        ## past 3 month 
        
        past_3month = end - relativedelta(months=3)
        back3Mdata = dflive.loc[(dflive.open_time> past_3month)]
        thmh = back3Mdata.h.max()
        thml = back3Mdata.l.min()
        
 
        tuple =(symbol,alth,altl,now,omh,oml,tmh,tml,thmh,thml,SMA21,SMA55,SMA100,)
        table.append(tuple)
        del dflive
        #print("name is: ",array[0] ,alth,altl)
        #print("name is: ",symbol)
        

table[0]
AllTime = pd.DataFrame(table,columns=['Indicator','ATH','ATL','Current','OMH','OML','TMH','TML','THMH','THML','SMA21','SMA55','SMA100'])
AllTime.ATH=AllTime.ATH.astype(float)
AllTime.ATL=AllTime.ATL.astype(float)
AllTime.Current=AllTime.Current.astype(float)
AllTime.OMH=AllTime.OMH.astype(float)
AllTime.OML=AllTime.OML.astype(float)
AllTime.TMH=AllTime.TMH.astype(float)
AllTime.TML=AllTime.TML.astype(float)
AllTime.THMH=AllTime.THMH.astype(float)
AllTime.THML=AllTime.THML.astype(float)
AllTime.dtypes

AllTime['ALLTIME']=(AllTime['Current']/AllTime['ATH'])*100
AllTime['MONTH']=(AllTime['Current']/AllTime['OMH'])*100
AllTime['2MONTH']=(AllTime['Current']/AllTime['TMH'])*100
AllTime['3MONTH']=(AllTime['Current']/AllTime['THMH'])*100
AllTime['SMADIFF']= abs(AllTime['SMA21']- AllTime['SMA55'])
AllTime['SMARATIO']= (AllTime['SMA21']/AllTime['SMA55'])
AllTime['GREEN'] = (AllTime['SMA21'])/2
AllTime['GREEN15'] = (AllTime['SMA21'])/1.5
AllTime['ORANGE'] = (AllTime['SMA21'])
AllTime['RED'] = (AllTime['SMA21'])*2
AllTime['CLOSERATIO']= (abs(AllTime['Current']- AllTime['GREEN']))/AllTime['GREEN']



AllTime.sort_values(by=['ALLTIME'],ascending=True,inplace = True)
#AllTime.sort_values(by='ATH',ascending=False)
#AllTime.to_csv("alltimeHL.csv", mode="w")
#data.sort_values("Name", axis = 0, ascending = True,inplace = True, na_position ='last')

## Display the Data 
AllTime


# In[59]:


#drop = AllTime.loc[(AllTime['RSI'] < 30)& (AllTime['STRSI'] < 0.1)]
#drop = AllTime.loc[(AllTime['SMARATIO'] <  1.05 ) &(AllTime['SMARATIO'] > 0.95)& (AllTime['ALLTIME'] <70) ]
#drop = AllTime.loc[(AllTime['Current'] <= AllTime['SMA21'])]
drop = AllTime.loc[(AllTime['Current'] <= AllTime['SMA21']/1.5)]

#drop = AllTime.loc[(AllTime['SMARATIO'] <  1 ) &(AllTime['SMARATIO'] > 0.95) & (AllTime['ALLTIME'] <70)]

#drop = AllTime.loc[(AllTime['MONTH'] < 80)&(AllTime[ '2MONTH'] < 70)&(AllTime['ALLTIME'] < 40)]
#drop = AllTime.loc[(AllTime['MONTH'] <70)&(AllTime['2MONTH'] > (AllTime['3MONTH']))]

sort = drop.sort_values(by=['CLOSERATIO'],ascending=False,inplace = True) 
#sort = drop.sort_values(by=['ALLTIME'],ascending=True,inplace = True) 

#drop = drop.loc[(drop['ALLTIME'] < drop['3MONTH'])]
display = drop[['Indicator','ATH','CLOSERATIO','RED','ORANGE','GREEN','Current']]
#print(drop)
save = drop[['Indicator','ATH','CLOSERATIO','RED','ORANGE','GREEN','Current']]
save=save.drop_duplicates(subset='Indicator',keep='last')

print(save)
print("Low EASY Band")
print(display.iloc[0:60,0:14])
print(display.iloc[60:110,])
print(display.iloc[110:160,])


# In[76]:


allbands = AllTime[['Indicator','RED','ORANGE','GREEN','Current']]
allbands.to_csv("allbands.csv")
allbands


# In[60]:


# red band 

## band to sell in down trends
redband = AllTime.loc[(AllTime['Current'] >= 2*AllTime['SMA21'])]
redband
redband.to_csv("redband.csv")


# In[77]:


## band to sell in down trends
orangeband = AllTime.loc[(AllTime['Current'] >= AllTime['SMA21'])]
test = orangeband[['Indicator','GREEN','ORANGE','RED','Current']]
test.to_csv("orangeband.csv")
test


# In[62]:


#easybandlist = display['Indicator']
#easybandlist


# In[63]:


save.to_csv("easyband.csv")
print(date.today)
print(date)
save.to_csv("easyband"+str(date.today())+".csv")
save


# In[64]:


#display.Current.astype(float)

display = display.loc[:] 
display['ORatio'] = display['ORANGE']/display['Current'] 
#display = display.loc[display['CLOSERATIO']> 0.01]
display

#display


# # entries

# In[65]:



#entrylist =[['ANKRUSDT',0.07684],['AVAXUSDT',57],['WAXPUSDT',0.3818],['REEFUSDT',0.01131],['LINKUSDT',14.15],['ARUSDT',30],['ATOMUSDT',23],['MIRUSDT',2.343],['ALGOUSDT',0.909],['ZRXUSDT',0.60],['XTZUSDT',2.856],['KSMUSDT',163],['NEARUSDT',14],['WINUSDT',0.0003138],['CELRUSDT',0.05063],['DENTUSDT',0.002459],['LRCUSDT',0.7047],['CRVUSDT',3.529],['IMXUSDT',2.709],['SCRTUSDT',5],['RENUSDT',0.2547],['SUSHIUSDT',4.522],['CAKEUSDT',11.24],['XLMUSDT',0.2109],['HNTUSDT',30],['OMGUSDT',4.5],['LINAUSDT',0.024],['MIRUSDT',2.343],['FLUXUSDT',1.523]]
entrylist =[['ANKRUSDT',0.05415],['AVAXUSDT',57],['WAXPUSDT',0.3174],['REEFUSDT',0.00806],['LINKUSDT',14.15],['ARUSDT',30],['ATOMUSDT',23],['MIRUSDT',2.343],['ALGOUSDT',0.7046],['ZRXUSDT',0.50],['XTZUSDT',2.856],['KSMUSDT',150],['NEARUSDT',10.320],['WINUSDT',0.0003138],['DENTUSDT',0.002459],['LRCUSDT',0.677],['CRVUSDT',2.12],['IMXUSDT',1.30],['SCRTUSDT',5],['RENUSDT',0.3047],['SUSHIUSDT',3.5],['CAKEUSDT',11.24],['XLMUSDT',0.2109],['HNTUSDT',22.7],['OMGUSDT',4.5],['LINAUSDT',0.01759],['FLUXUSDT',1.523],['SRMUSDT',1.8],['VTHOUSDT',0.0031],['ANTUSDT',5.616],['RAYUSDT',2.5],['ROSEUSDT',0.24055],['MOVRUSDT',51.6],['TFUELUSDT',0.18],['GLMRUSDT',4.5],['COTIUSDT',0.2277],['ONEUSDT',0.13171],['VETUSDT',0.05138],['RUNEUSDT',3.709],['CELRUSDT',0.034063],['ONTUSDT',0.4652],['GALAUSDT',0.24794],['GRTUSDT',0.3164],['DOTUSDT',16.31],['ZILUSDT',0.04410],['FILUSDT',20],['SCUSDT',0.011],['MATICUSDT',1.4],['FTMUSDT',1.15],['BANDUSDT',3.6],['1INCHUSDT',1.6],['AUDIOUSDT',1.0],['IOTXUSDT','0.07109'],['CTSIUSDT','0.62'],['LUNAUSDT','13'],['SOLUSDT','35']]
exposurelist =[['ANKRUSDT',311],['AVAXUSDT',2850],['WAXPUSDT',527],['REEFUSDT',700],['LINKUSDT',6000],['ARUSDT',2450],['ATOMUSDT',1863],['MIRUSDT',85],['ALGOUSDT',3300],['ZRXUSDT',245],['XTZUSDT',1200],['KSMUSDT',7800],['NEARUSDT',1360],['WINUSDT',285],['DENTUSDT',385],['LRCUSDT',660],['CRVUSDT',323],['IMXUSDT',269],['SCRTUSDT',601],['RENUSDT',684],['SUSHIUSDT',2730],['CAKEUSDT',4000],['XLMUSDT',368],['HNTUSDT',1250],['OMGUSDT',765],['LINAUSDT',205],['FLUXUSDT',224],['SRMUSDT',587],['VTHOUSDT',413],['ANTUSDT',200],['RAYUSDT',100],['ROSEUSDT',634],['MOVRUSDT',500],['TFUELUSDT',344],['GLMRUSDT',400],['COTIUSDT',500],['ONEUSDT',1000],['VETUSDT',500],['RUNEUSDT',750],['CELRUSDT',500],['ONTUSDT',400],['GALAUSDT',650],['GRTUSDT',1700],['DOTUSDT',10000],['ZILUSDT',1000],['FILUSDT',2000],['SCUSDT',200],['MATICUSDT',8000],['FTMUSDT',2000],['BANDUSDT',750],['1INCHUSDT',1000],['AUDIOUSDT',700],['IOTXUSDT',500],['CTSIUSDT',325],['LUNAUSDT','4000'],['SOLUSDT','3000'],['DOTUSDT','10000']]
startdate = '2021-01-01'
entries = pd.DataFrame(entrylist, columns =['Indicator', 'lowest_entry'],dtype = float) 
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
exposure.to_csv("exposure.csv")

entries = entries.merge(exposure ,how="left",on="Indicator")
#entries

fixentries = entries.merge(display ,how='left',on=['Indicator'])
#fixentries = fixentries.merge(exposure,how='left',on=['Indicator'])
#fixentries
#entries1 = entries.merge(fixentries,how='left',on=['Indicator'])
fixentries = fixentries.dropna(axis=0,subset=['ATH'])
fixentries=fixentries.drop_duplicates(subset='Indicator',keep='last')
fixentries.lowest_entry.astype(float)
fixentries.Current.astype(float)


fixentries['DROP'] = ((fixentries.lowest_entry - fixentries.Current)/fixentries.lowest_entry)*100
fixentries.sort_values(by=['DROP'],ascending=False,inplace = True)
fixentries


# # Red Band

# In[66]:


## over red entries 
overred = redband.merge(entries ,how="left",on="Indicator")
overed = overred.dropna(axis=0,subset=['RED'])
overred= overred[['Indicator','GREEN','ORANGE','RED','Current']]
overred


# # Orange Band

# In[67]:


## over orange entries 
overorange = entries.merge(orangeband ,how="left",on="Indicator")
overorange = overorange.dropna(axis=0,subset=['GREEN'])
overoranged= overorange[['Indicator','GREEN','ORANGE','RED','Current']]
overoranged


# In[68]:


save = fixentries
save.to_csv("fixoneasy.csv")
save
del fixentries


# In[69]:


#Gap is an acceptable area for rebounce and exit
orangedrop = save.loc[:]
orangedrop['GapOrange']= orangedrop['ORANGE']/orangedrop['Current']
orangedrop['GapRed']= orangedrop['RED']/orangedrop['Current']
orangedrop['GapATH']= orangedrop['ATH']/orangedrop['Current']
#orangedrop['GapAORT']= (orangedrop['GapATH']*orangedrop['GapOrange'])


orangedrop


# In[70]:


# multiplying the gap with the value of the drop for a reasonable recovery
exposuredrop = orangedrop
exposuredrop['Total']= exposuredrop['exposure']*exposuredrop['DROP']*exposuredrop['GapOrange']
#exposuredrop['Total']= exposuredrop['DROP']*exposuredrop['GapOrange']*exposuredrop['CLOSERATIO']
#exposuredrop['Total']= exposuredrop['GapOrange']*exposuredrop['DROP']
#exposuredrop = exposuredrop.loc[exposuredrop['DROP']>0]

#exposuredrop.sort_values(by=['GapOrange'],ascending=False,inplace = True)
exposuredrop.sort_values(by=['CLOSERATIO'],ascending=False,inplace = True)

exposuredrop


# In[71]:


exposuredrop.to_csv("exposuredrop.csv")


# In[72]:


# easy band and cap ratio
capratio = pd.read_csv('capratio.csv')
capratio = pd.DataFrame(capratio,columns=['Indicator','supply_ratio','cap_ratio'])
capratio

exposuredrop = exposuredrop.merge(capratio ,how='left',on=['Indicator'])
sort = exposuredrop.sort_values(by=['Total'],ascending=False,inplace = True)

exposuredrop


# In[73]:


undergreen = exposuredrop.loc[exposuredrop['Current'] < exposuredrop['GREEN']]
undergreen


# In[ ]:





# In[ ]:




