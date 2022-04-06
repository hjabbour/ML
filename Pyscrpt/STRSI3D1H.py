#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
from datetime import date
#Data Source (Pandas Data Reader)
from pandas_datareader import data
import requests 
import json
from dateutil.relativedelta import relativedelta


#Data Source(Pandas Data Reader)


# In[3]:


## choose either white list or read from fix entries 

#fixentries= pd.read_csv('easyband.csv')
#fixentries= pd.read_csv('fixoneasy.csv')
#fixentries= pd.read_csv('fixentries.csv')
fixentries= pd.read_csv('entries.csv')


fixentries
whitelist = []
whitelist = fixentries['Indicator'].array
whitelist[0]
#final = pd.DataFrame(fixentries['Indicator']) 
#final.set_index('Indicator')


# In[4]:


#alllist = 'GALA/USDT, ANT/USDT,IMX/USDT, MATIC/USDT, ZRX/USDT,  ALGO/USDT, BTC/USDT, ETH/USDT, AXS/USDT, SOL/USDT, MATIC/USDT, ADA/USDT, BNB/USDT, FIL/USDT, XRP/USDT, CVC/USDT, FTM/USDT, ICP/USDT, DOGE/USDT, XEC/USDT, IOTA/USDT, AVAX/USDT, DOT/USDT, LTC/USDT, EOS/USDT, IOST/USDT, STMX/USDT, FTT/USDT, TWT/USDT, ALICE/USDT, ATA/USDT, ETC/USDT, VET/USDT, SAND/USDT, LINK/USDT, THETA/USDT, SHIB/USDT, CHZ/USDT, LUNA/USDT, IOTX/USDT, TRX/USDT, TLM/USDT, BCH/USDT, REEF/USDT, CAKE/USDT, NEO/USDT, SLP/USDT, RVN/USDT, ATOM/USDT, SUSHI/USDT, ARDR/USDT, OMG/USDT, GRT/USDT, 1INCH/USDT, AAVE/USDT, C98/USDT, UNI/USDT, HBAR/USDT, CRV/USDT, DENT/USDT, MBOX/USDT, BTT/USDT, SRM/USDT, YFI/USDT, HOT/USDT, XLM/USDT, FIS/USDT, REQ/USDT, SXP/USDT, KSM/USDT, QNT/USDT, NEAR/USDT, COMP/USDT, QTUM/USDT, CHR/USDT, SC/USDT, XVS/USDT, ONT/USDT, COTI/USDT, NKN/USDT, RUNE/USDT, ANKR/USDT, MANA/USDT, CTXC/USDT, AR/USDT, RAY/USDT, BAKE/USDT, ARPA/USDT, XTZ/USDT, ZIL/USDT, ONG/USDT, ROSE/USDT, PNT/USDT, ALGO/USDT, MINA/USDT, WIN/USDT, AUDIO/USDT, STORJ/USDT, EPS/USDT, MDX/USDT, ZEC/USDT, ENJ/USDT, HIVE/USDT, TFUEL/USDT, WRX/USDT, SNX/USDT, SKL/USDT, LINA/USDT, DODO/USDT, UNFI/USDT, XEM/USDT, EGLD/USDT, ICX/USDT, WAVES/USDT, ONE/USDT, DASH/USDT, KAVA/USDT, SUPER/USDT, TKO/USDT, OGN/USDT, XMR/USDT, OCEAN/USDT, CELO/USDT, BAT/USDT, FUN/USDT, LRC/USDT, ALPHA/USDT, MASK/USDT, TRU/USDT, DNT/USDT, LIT/USDT, FET/USDT, GTC/USDT, CELR/USDT, DEXE/USDT, ZEN/USDT, WAXP/USDT, CTSI/USDT, HNT/USDT, YFII/USDT, BEL/USDT, ZRX/USDT, DEGO/USDT, RSR/USDT, COS/USDT, FLM/USDT, BAL/USDT, MFT/USDT, RLC/USDT, KNC/USDT, REN/USDT, INJ/USDT, FLOW/USDT, SFP/USDT, BAND/USDT, CTK/USDT, SUN/USDT, AKRO/USDT, XVG/USDT, NANO/USDT, STRAX/USDT, PUNDIX/USDT, TVK/USDT, MKR/USDT, CLV/USDT, TRB/USDT, ETHUP/USDT, KEEP/USDT, TCT/USDT, POND/USDT, LTO/USDT, ANT/USDT, TOMO/USDT, ACM/USDT, FIO/USDT, VITE/USDT, MBL/USDT, JST/USDT, MTL/USDT, UTK/USDT, BLZ/USDT, ORN/USDT, STPT/USDT, CFX/USDT, OM/USDT, PERP/USDT, MIR/USDT, STX/USDT, ERN/USDT, TUSD/USDT, PSG/USDT, NULS/USDT, OXT/USDT, REP/USDT, BNBUP/USDT, DATA/USDT, WAN/USDT, BURGER/USDT, RAMP/USDT, ATM/USDT, DGB/USDT, DIA/USDT, BEAM/USDT, UMA/USDT, ETHDOWN/USDT, CKB/USDT, TRIBE/USDT, FARM/USDT, BTCUP/USDT, BZRX/USDT, ALPACA/USDT, DOTUP/USDT, BTS/USDT, FOR/USDT, ADAUP/USDT, HARD/USDT, LSK/USDT, POLS/USDT, FORTH/USDT, TORN/USDT, VTHO/USDT, BTG/USDT, QUICK/USDT, FILUP/USDT, PAX/USDT, MITH/USDT, SUSHIUP/USDT, BTCDOWN/USDT, XRPDOWN/USDT, BNBDOWN/USDT, DUSK/USDT, XRPUP/USDT, WTC/USDT, DOCK/USDT, AUTO/USDT, BTCST/USDT, KLAY/USDT, WNXM/USDT, KEY/USDT, FILDOWN/USDT, NU/USDT, IRIS/USDT, KMD/USDT, TROY/USDT, WING/USDT, AION/USDT, PHA/USDT, SUSHIDOWN/USDT, EOSUP/USDT, LTCUP/USDT, GHST/USDT, PAXG/USDT, BNT/USDT, AVA/USDT, BADGER/USDT, PERL/USDT, COCOS/USDT, GXS/USDT, MDT/USDT, 1INCHUP/USDT, GTO/USDT, LINKUP/USDT, AAVEUP/USDT, FIRO/USDT, JUV/USDT, MLN/USDT, UNIUP/USDT, LPT/USDT, NMR/USDT, AAVEDOWN/USDT, 1INCHDOWN/USDT, ASR/USDT, OG/USDT, BAR/USDT, DREP/USDT, ADADOWN/USDT, BOND/USDT, DCR/USDT, DOTDOWN/USDT, GNO/USDT, NBS/USDT, SXPUP/USDT, EOSDOWN/USDT, YFIUP/USDT, RIF/USDT, LTCDOWN/USDT, XLMUP/USDT, YFIDOWN/USDT, LINKDOWN/USDT, TRXUP/USDT, BCHUP/USDT, XTZUP/USDT, SXPDOWN/USDT, UNIDOWN/USDT, XTZDOWN/USDT, BCHDOWN/USDT, TRXDOWN/USDT, XLMDOWN/USDT, SUSD/USDT'
#whitelist = 'LINK/USDT, DOT/USDT'
#whitelist =str(fixentries['Indicator'])

startdate = '2020-01-01'
enddate = pd.to_datetime('today')
intervals = ['1h','2h','4h','8h','12h','1d','3d']
#intervals = ['8h']


# In[5]:


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


# In[6]:


#alllist=alllist.lstrip()
#array = alllist.split(",")
#array = whitelist
array = whitelist
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

#print(intervals)
## get the data 
import os

live = pd.to_datetime('today')
start = pd.to_datetime(startdate)
end = pd.to_datetime(enddate)

root_url = 'https://api.binance.com/api/v1/klines'
#symbol = 'STEEMETH'
#interval = '1d'
#url = root_url + '?symbol=' + symbol + '&interval=' + interval
#print(url)


directory = 'BN-main'
table = []
interval = ''
final = pd.DataFrame(symbols,columns =['Indicator'])
#final.set_index('Indicator')

for interval in intervals:
    

    for symbol in symbols:
            url = root_url + '?symbol=' + symbol + '&interval=' + interval
            data = json.loads(requests.get(url).text)
            dflive = pd.DataFrame(data ,columns =['open_time',
                  'o', 'h', 'l', 'c', 'v',
                  'close_time', 'qav', 'num_trades',
                  'taker_base_vol', 'taker_quote_vol', 'ignore'])
            dflive.open_time = pd.to_datetime(dflive['open_time'],unit='ms')
            dflive=dflive.loc[(dflive.open_time>start)]
            
            if dflive.size > 300 :
                
                
                

                dflive = dflive.astype ({'o': float ,'h':float,'l':float ,'c':float ,'v':float})
                dflive['RSI'] = FRSI(dflive['c'])
                dflive['STRSI'] = STRSIA(dflive['c'])[0]
                dflive['STRSIK'] = STRSIA(dflive['c'])[1]
                dflive['STRSID'] = STRSIA(dflive['c'])[2]



                #print(dflive)


                lastrow = dflive.iloc[-1]
                beforelast = dflive.iloc[-2]
                now = float(lastrow.c)
                RSI = float(lastrow.RSI)
                RSIM = float(beforelast.RSI)
                STRSI = float(lastrow.STRSI)
                STRSIK = float(lastrow.STRSIK)
                STRSIKM = float(beforelast.STRSIK)
                STRSID = float(lastrow.STRSID)
                STRSIDM = float(beforelast.STRSID)


                tuple =(symbol,RSI,RSIM,STRSI,STRSIK,STRSIKM,STRSID,STRSIDM)
                table.append(tuple)
                #print(table)
                del dflive
            else:
                pass
            #print("name is: ",array[0])


    table[0]
    AllTime = pd.DataFrame(table,columns=['Indicator','RSI','RSIM','STRSI','STRSIK','STRSIKM','STRSID','STRSIDM'])


    #AllTime.Current=AllTime.Current.astype(float)

    AllTime.dtypes


    #AllTime.sort_values(by='ATH',ascending=False)
    #AllTime.to_csv("alltimeHL.csv", mode="w")
    #data.sort_values("Name", axis = 0, ascending = True,inplace = True, na_position ='last')

    ## Display the Data 

    test=AllTime.rename(columns={'RSI':('RSI'+interval),'RSIM':('RSIM'+interval),'STRSI':('STRSI'+interval),'STRSIK':('STRSIK'+interval),'STRSIKM':('STRSIKM'+interval),'STRSID':('STRSID'+interval),'STRSIDM':('STRSIDM'+interval)})
    test.set_index(['Indicator'])
    final=final.merge(test,how='left',on=['Indicator'])
            

final


# In[7]:


#final.Indicator


# In[8]:


final=final.drop_duplicates(subset='Indicator',keep='last')
#final= final.drop(columns=['Current_x','Current_y'])
final.to_csv("STRSI3D1H.csv")
final


# In[9]:


#Rules

rule12under = (final['STRSIK12h'] < final['STRSID12h'])
rule12over = (final['STRSIK12h'] > final['STRSID12h'])
rule12kdown = (final['STRSIK12h'] < final['STRSIKM12h'])
rule12kup = (final['STRSIK12h'] > final['STRSIKM12h'])
rule12RSIup = (final['RSI12h'] > final['RSIM12h'])
rule12RSIdown = (final['RSI12h'] < final['RSIM12h'])

bottom12h = (final['STRSIK12h']< 0.2)
bottomr12h = (final['RSI12h'] < 30)
top12h = (final['STRSIK12h'] > 0.8)
topr12h = (final['RSI12h'] > 70)
room12h = (final['STRSIK12h'] < 0.8)
roomr12h = (final['RSI12h'] < 70)

shift12h = (rule12under & (rule12kup | rule12RSIup))
dshift12h = (rule12over & (rule12kdown | rule12RSIdown))
uptrend12h = (rule12over & (rule12kup & rule12RSIup))
downtrend12h = (rule12under & (rule12kdown & rule12RSIdown))





rule8under = (final['STRSIK8h']< final['STRSID8h'])
rule8over = (final['STRSIK8h'] > final['STRSID8h'])
rule8kdown = (final['STRSIK8h']< final['STRSIKM8h'])
rule8kup = (final['STRSIK8h'] > final['STRSIKM8h'])
rule8RSIup = (final['RSI8h'] > final['RSIM8h'])
rule8RSIdown = (final['RSI8h'] < final['RSIM8h'])

bottom8h = (final['STRSIK8h']< 0.2)
bottomr8h = (final['RSI8h'] < 30)
top8h = (final['STRSIK8h'] > 0.8)
topr8h = (final['RSI8h'] > 70)
room8h = (final['STRSIK8h'] < 0.8)
roomr8h = (final['RSI8h'] < 70)

shift8h = (rule8under & (rule8kup | rule8RSIup))
dshift8h = (rule8over & (rule8kdown | rule8RSIdown))
uptrend8h = (rule8over & (rule8kup & rule8RSIup))
downtrend8h = (rule8under & (rule8kdown & rule8RSIdown))

rule4under = (final['STRSIK4h']< final['STRSID4h'])
rule4over = (final['STRSIK4h'] > final['STRSID4h'])
rule4kdown = (final['STRSIK4h']< final['STRSIKM4h'])
rule4kup = (final['STRSIK4h'] > final['STRSIKM4h'])
rule4RSIup = (final['RSI4h'] > final['RSIM4h'])
rule4RSIdown = (final['RSI4h'] < final['RSIM4h'])

bottom4h = (final['STRSIK4h']< 0.2)
bottomr4h = (final['RSI4h'] < 30)
top4h = (final['STRSIK4h'] > 0.8)
topr4h = (final['RSI4h'] > 70)
room4h = (final['STRSIK4h'] < 0.8)
roomr4h = (final['RSI4h'] < 70)

shift4h = (rule4under & (rule4kup | rule4RSIup))
dshift4h = (rule4over & (rule4kdown | rule4RSIdown))
uptrend4h = (rule4over & (rule4kup & rule4RSIup))
downtrend4h = (rule4under & (rule4kdown & rule4RSIdown))




rule2under = (final['STRSIK2h']< final['STRSID2h'])
rule2over = (final['STRSIK2h'] > final['STRSID2h'])
rule2kdown = (final['STRSIK2h']< final['STRSIKM2h'])
rule2kup = (final['STRSIK2h'] > final['STRSIKM2h'])
rule2RSIup = (final['RSI2h'] > final['RSIM2h'])
rule2RSIdown = (final['RSI2h'] < final['RSIM2h'])

bottom2h = (final['STRSIK2h']< 0.2)
bottomr2h = (final['RSI2h'] < 30)
top2h = (final['STRSIK2h'] > 0.8)
topr2h = (final['RSI2h'] > 70)
room2h = (final['STRSIK2h'] < 0.8)
roomr2h = (final['RSI2h'] < 70)


shift2h = (rule2under & (rule2kup | rule2RSIup))
dshift2h = (rule2over & (rule2kdown | rule2RSIdown))
uptrend2h = (rule2over & (rule2kup & rule2RSIup))
downtrend2h = (rule2under & (rule2kdown & rule2RSIdown))


rule1under = (final['STRSIK1h']< final['STRSID1h'])
rule1over = (final['STRSIK1h'] > final['STRSID1h'])
rule1kdown = (final['STRSIK1h']< final['STRSIKM1h'])
rule1kup = (final['STRSIK1h'] > final['STRSIKM1h'])
rule1RSIup = (final['RSI1h'] > final['RSIM1h'])
rule1RSIdown = (final['RSI1h'] < final['RSIM1h'])

bottom1h = (final['STRSIK1h']< 0.2)
bottomr1h = (final['RSI1h'] < 30)
top1h = (final['STRSIK1h'] > 0.8)
topr1h = (final['RSI1h'] > 70)
room1h = (final['STRSIK1h'] < 0.8)
roomr1h = (final['RSI1h'] < 70)


shift1h = (rule1under & (rule1kup | rule1RSIup))
dshift1h = (rule1over & (rule1kdown | rule1RSIdown))
uptrend1h = (rule1over & (rule1kup & rule1RSIup))
downtrend1h = (rule1under & (rule1kdown & rule1RSIdown))



rule1dunder = (final['STRSIK1d']< final['STRSID1d'])
rule1dover = (final['STRSIK1d'] > final['STRSID1d'])
rule1dkdown = (final['STRSIK1d']< final['STRSIKM1d'])
rule1dkup = (final['STRSIK1d'] > final['STRSIKM1d'])
rule1dRSIup = (final['RSI1d'] > final['RSIM1d'])
rule1dRSIdown = (final['RSI1d'] < final['RSIM1d'])

bottom1d = (final['STRSIK1d']< 0.2)
bottomr1d = (final['RSI1d'] < 30)
top1d = (final['STRSIK1d'] > 0.8)
topr1d = (final['RSI1d'] > 70)
room1d = (final['STRSIK1d'] < 0.8)
roomr1d = (final['RSI1d'] < 70)

shift1d = (rule1dunder & (rule1dkup | rule1dRSIup))
dshift1d = (rule1dover & (rule1dkdown | rule1dRSIdown))
uptrend1d = (rule1dover & (rule1dkup & rule1dRSIup))
downtrend1d = (rule1dunder & (rule1dkdown & rule1dRSIdown))


rule3dunder = (final['STRSIK3d']< final['STRSID3d'])
rule3dover = (final['STRSIK3d'] > final['STRSID3d'])
rule3dkdown = (final['STRSIK3d']< final['STRSIKM3d'])
rule3dkup = (final['STRSIK3d'] > final['STRSIKM3d'])
rule3dRSIup = (final['RSI3d'] > final['RSIM3d'])
rule3dRSIdown = (final['RSI3d'] < final['RSIM3d'])

bottom3d = (final['STRSIK3d']< 0.2)
bottomr3d = (final['RSI3d'] < 30)
top3d = (final['STRSIK3d'] > 0.8)
topr3d = (final['RSI3d'] > 70)
room3d = (final['STRSIK3d'] < 0.8)
roomr3d = (final['RSI3d'] < 70)


shift3d = (rule3dunder & (rule3dkup | rule3dRSIup))
dshift3d = (rule3dover & (rule3dkdown | rule3dRSIdown))
uptrend3d = (rule3dover & (rule3dkup & rule3dRSIup))
downtrend3d = (rule3dunder & (rule3dkdown & rule3dRSIdown))




rule13over = (final['STRSIK3d'] > final['STRSID3d']) & (final['STRSIK1d'] > final['STRSID1d'])
rule13under =(final['STRSIK3d'] < final['STRSID3d']) & (final['STRSIK1d'] < final['STRSID1d'])




final.dropna()


# # Summary  draft rules
# 

# In[10]:


drop = final.loc[bottom4h & (shift4h | rule4over)]
drop = final.loc[(bottom4h | (uptrend4h | shift4h)) & (uptrend8h | shift8h) ]
#drop = final.loc[((uptrend4h | shift4h)) & (uptrend8h | shift8h) ]
#drop = final.loc[(uptrend8h | shift8h) & (uptrend12h | shift12h) & (uptrend1d | shift1d)]
#drop = final.loc[(uptrend2h | shift2h) & (uptrend4h | shift4h) ]
#drop = final.loc[(uptrend4h | shift4h)]

#drop = final.loc[bottom8h | (shift8h | rule8over)]
#drop = final.loc[bottom12h | (shift12h | rule12over)]
#drop = final.loc[(uptrend8h)]
#drop = final.loc[bottom12h | (shift12h | uptrend12h)]
#drop = final.loc[bottom1d | (shift1d | uptrend1d)]
#drop = final.loc[top4h]
#drop = final.loc[(uptrend3d)]
#drop = final.loc[dshift4h]
#drop = final.loc[shift2h]

#drop = final.loc[(bottom4h & bottom12h & bottom3d)]
#drop = final.loc[(uptrend8h)]
#drop = final.loc[(downtrend8h & downtrend4h & downtrend12h)]
#drop = final.loc[(bottom4h & bottom2h & bottom3d)]
#drop = final.loc[shift3d & bottom4h]

#drop = final.loc[(bottom12h)]
#drop = final.loc[ ((rule4kup |rule4RSIup))]

#drop = final.loc[((uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend4h | shift4h) ]
#drop = final.loc[((uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) ]

# 4 hours will go down or slow down
#drop = final.loc[( (bottom3d) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) & dshift4h) ]
#drop = final.loc[( (bottom3d) & (uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (uptrend4h | shift4h) ]
#drop = final.loc[( (bottom3d) & (uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (uptrend4h | shift4h) &  (uptrend2h | shift2h) &  (uptrend1h | shift1h)]

#drop = final.loc[( (uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (uptrend4h | shift4h) ]

drop = final.loc[( (bottom4h) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h)) ]
#drop = final.loc[((bottom1d) & (bottom3d))]
#drop = drop.loc[( (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h))]
#drop = final.loc[dshift4h]

#drop = final.loc[( (uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (uptrend4h | shift4h) & (downtrend2h |dshift2h) ]
#drop = final.loc[( (bottom3d) & (uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (uptrend4h | shift4h) ]
#drop = final.loc[dshift2h]
#drop = final.loc[( (bottom8h) & (bottom12h) & (bottom4h) & (bottom2h))]
#drop = final.loc[( (bottom8h) & (bottom12h) & (bottom4h))]

#drop= final.loc[(shift4h) & (shift12h)]
#drop = final.loc[(dshift4h | downtrend4h)]
#drop = final.loc[( (bottom4h)&(bottom8h) & (bottom12h) & (bottom1d) & (dshift4h | downtrend4h))]
#drop = final.loc[( (bottom4h)&(bottom8h) & (bottom12h) & (bottom1d) & (shift2h))]

#drop = final.loc[downtrend1d & downtrend12h & downtrend8h & downtrend4h]
#drop = final.loc[top8h]

drop


# # Bottoms
# 

# In[11]:


drop = final.loc[((bottom12h) & (bottom1d) & (bottom3d))]
#drop = final.loc[( (bottom4h)&(bottom8h) & (bottom12h) & (bottom1d) & (bottom3d))]
drop = final.loc[( (bottom8h) & (bottom12h) & (bottom1d))]
#drop = final.loc[( (bottom8h) & (bottom12h))]
drop = final.loc[( (bottom8h) & (bottom2h) & (bottom4h) & (bottom1h))]
#drop = final.loc[( (bottom2h) & (bottom4h) & (bottom1h) & (bottomr1h))]
#drop = final.loc[( (bottom2h) & (bottom4h) & (bottom1h))]

#drop = final.loc[( (bottom2h) & (bottom1h))]
#drop = final.loc[( (bottom1h))]
#drop = final.loc[((bottom1d) & (bottom3d))]
#drop = final.loc[((bottom1d))]
#drop = final.loc[((bottom3d))]

#drop = drop.loc[(shift8h)]

drop


# In[12]:


# Bottom RSI

drop = final.loc[((bottomr12h) & (bottomr1d) & (bottomr3d))]

drop = final.loc[((bottomr12h) & (bottomr1d) & (bottomr3d))]
#drop = final.loc[( (bottomr4h)&(bottomr8h) & (bottomr12h) & (bottomr1d) & (bottomr3d))]
#drop = final.loc[( (bottomr8h) & (bottomr12h) & (bottomr1d))]
#drop = final.loc[( (bottomr8h) & (bottomr12h))]
#drop = final.loc[( (bottomr8h) & (bottomr2h) & (bottomr4h) & (bottomr1h))]
#drop = final.loc[( (bottomr8h) & (bottomr2h) & (bottomr4h))]
#drop = final.loc[( (bottomr2h) & (bottomr4h) & (bottomr1h))]
#drop = final.loc[( (bottomr2h) & (bottomr4h))]
drop = final.loc[( (bottomr2h) & (bottomr1h))]
#drop = final.loc[( (bottomr1h))]
#drop = final.loc[((bottomr1d) & (bottomr3d))]
#drop = final.loc[((bottomr2h))]
#drop = final.loc[((bottomr8h))]



drop


# # Tops
# 

# In[13]:


drop = final.loc[((top1d) & (top3d))]

drop = final.loc[( (top4h) & (top2h) & (top1h) & (top8h) & (top12h) & (top1d) & (top3d))]
drop = final.loc[( (top8h) & (top12h) & (top1d) & (top3d))]
drop = final.loc[( (top8h) & (top4h) & (top2h))]
#drop = final.loc[( (top4h) & (top2h) & (top1h))]
#drop = final.loc[((top4h) & (top2h))]
#drop = final.loc[((top1d) & (top3d))]


drop


# In[14]:


#TOP R
drop = final.loc[((topr1d) & (topr3d))]

drop = final.loc[( (topr4h) & (topr2h) & (topr1h) & (topr8h) & (topr12h) & (topr1d) & (topr3d))]
drop = final.loc[( (topr8h) & (topr12h) & (topr1d) & (topr3d))]
#drop = final.loc[( (topr8h) & (topr4h) & (topr2h))]
#drop = final.loc[( (topr4h) & (topr2h) & (topr1h))]
drop = final.loc[((topr4h) & (topr2h))]
#drop = final.loc[( (topr2h) & (topr1h))]
#drop = final.loc[((topr1h))]


drop


# # UP Trend

# In[15]:


#drop = final.loc[( (bottom3d) & (uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (uptrend4h | shift4h) ]

drop = final.loc[( (uptrend3d)) & (uptrend8h) & (uptrend1d) & (uptrend12h ) &  (uptrend4h) ]
#drop = final.loc[( (uptrend3d) & (uptrend1d)) ]
#drop = final.loc[( (uptrend3d) & (uptrend1d) & (uptrend12h )) ]
#drop = final.loc[( (uptrend3d) & (uptrend1d) & (uptrend12h ) & (uptrend8h)) ]
#drop = drop.loc[(room3d)&(room1d)&(room12h)]
#drop = drop.loc[(roomr1d & roomr3d & roomr12h)]
#drop = final.loc[(roomr1d & room1d)]

drop


# # UP with OR 

# In[16]:


## UP with K or RSI might show early reversal 
#drop = final.loc[((rule3dkup | rule3dRSIup) & (rule1dkup | rule1dRSIup) & (rule12kup & rule12RSIup) & (rule8kup & rule8RSIup) & (rule4kup | rule4RSIup) & (final['STRSIK1d'] < 0.2) ) ]
#drop = final.loc[((rule3dkup | rule3dRSIup) & (rule1dkup | rule1dRSIup) & (rule12kup & rule12RSIup) & (rule8kup & rule8RSIup) & (rule4kup | rule4RSIup)  ) ]

#drop = final.loc[((rule3dkup | rule3dRSIup) & (rule1dkup | rule1dRSIup) & (rule12kup & rule12RSIup) & (rule8kup & rule8RSIup) & (rule4kup | rule4RSIup) &  (rule2kup | rule2RSIup) & (rule1kup | rule1RSIup) ) ]
#drop = final.loc[((rule3dkup | rule3dRSIup) & (rule1dkup | rule1dRSIup) & (rule12kup & rule12RSIup) & (rule8kup & rule8RSIup) & (rule4kup | rule4RSIup) &  (rule2kup | rule2RSIup) & (rule1kup | rule1RSIup) ) ]

#drop = final.loc[((uptrend3d) & (uptrend1d) & (uptrend12h) & (uptrend8h)) ]

#drop


# # Up Shifts
# 

# In[17]:


# 3d to 8 hours with room
drop = final.loc[( (bottom3d) & (uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (uptrend4h | shift4h) ]
drop = final.loc[((uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (uptrend4h | shift4h) ]
drop = final.loc[((room1d & room12h & room8h & roomr3d & roomr1d & roomr12h & roomr8h) &(uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h)]
#drop = final.loc[((room3d) &(uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h)]

#drop = final.loc[((room3d & room1d & room12h & room8h & room4h & room2h) &(uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (uptrend4h | shift4h) ]
#drop = final.loc[((room3d & room1d & room12h & room8h & room4h & room2h & room1h) &(uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) ]

#drop = final.loc[((uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (uptrend4h | shift4h) & (uptrend2h | shift2h) &  (uptrend1h | shift1h)]

#drop = final.loc[((uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h)]
#drop = final.loc[( (uptrend3d | shift3d) & (uptrend1d | shift1d) ) ]

#drop = drop.loc[((downtrend2h | dshift2h) | (downtrend1h | dshift1h) )]
drop


# In[18]:


# UP alone shift alone
drop = final.loc[((uptrend3d)) & (uptrend8h) & (uptrend1d) & (uptrend12h) &  (uptrend4h) ]
#drop = final.loc[(shift3d) & (shift8h) & (shift1d) & (shift12h) &  (shift4h) ]
drop = final.loc[(shift8h) & (shift12h) &  (shift4h) ]
drop = final.loc[(shift8h) & (shift12h) ]
#drop = final.loc[(shift8h)]
#drop = final.loc[(shift4h)]


#drop = final.loc[(shift8h) & (shift4h) ]


drop


# In[19]:


# up with room till 12h
drop = final.loc[((roomr3d & roomr1d & roomr12h & room12h ) &(uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h)]
drop


# In[20]:


# dshift on 4
drop = final.loc[(downtrend4h | dshift4h)]
drop


# In[21]:


# dshift on 2 
drop = final.loc[(downtrend2h | dshift2h)]
drop


# In[22]:


# UP with room R till 8 hours
drop = final.loc[((room12h & roomr3d & roomr1d & roomr12h & roomr8h) &(uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (uptrend4h | shift4h) ]
drop


# In[23]:


# UP with room till 2 hours
drop = final.loc[((room3d & room1d & room12h & room8h & room4h & room2h) &(uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (uptrend4h | shift4h) ]
drop


# In[24]:


# UP trend shift 
#drop = final.loc[((downtrend2h | dshift2h) & (downtrend1h | dshift1h) )]
#drop = final.loc[((uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (uptrend4h | shift4h) ]
drop = final.loc[((uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) ]

#drop = final.loc[((uptrend3d | shift3d)) & (uptrend1d | shift1d) & (uptrend12h | shift12h) ]

#drop = drop.loc[((downtrend2h | dshift2h) & (downtrend1h | dshift1h) )]
#drop = drop.loc[((downtrend2h | dshift2h))]
#drop = drop.loc[((downtrend2h | dshift2h))]

drop


# In[25]:


# UP with top 8 and 4 
drop = final.loc[((uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (uptrend4h | shift4h) ]
drop = drop.loc[((top8h & top4h) &(uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (uptrend4h | shift4h) ]
drop


# In[26]:


# UP 3d to 8 hours and 4 hours down 
drop = final.loc[((uptrend3d | shift3d)) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h) &  (dshift4h | downtrend4h) ]
drop


# # Down options

# In[27]:


#
drop = final.loc[(downtrend3d) & (downtrend8h) & (downtrend1d) & (downtrend12h) &  (downtrend4h) & (downtrend2h) & (downtrend1h)]
drop = final.loc[((downtrend3d)) & (downtrend8h) & (downtrend1d) & (downtrend12h) &  (downtrend4h) ]
#drop = final.loc[ (downtrend8h)  &  (downtrend4h) & (downtrend2h) & (downtrend1h)]
#drop = final.loc[ (downtrend4h) & (downtrend2h) & (downtrend1h)]

#drop = final.loc[((dshift3d)) & (dshift8h) & (dshift1d) & (dshift12h) &  (dshift4h) ]

#drop = final.loc[((downtrend3d | dshift3d)) & (downtrend8h | dshift8h) & (downtrend1d | dshift1d) & (downtrend12h | dshift12h) &  (downtrend4h | dshift4h) ]
drop


# In[28]:


## 3 Days going down
drop = final.loc[((rule3dkdown & rule3dRSIdown))]
drop


# In[29]:


## one Day down

drop = final.loc[((rule1dkdown & rule1dRSIdown))]
drop


# In[30]:


## FULL down options from 1 day till 4 hours 

drop = final.loc[((rule1dkdown & rule1dRSIdown) & (rule12kdown & rule12RSIdown) & (rule8kdown & rule8RSIdown) & (rule4kdown & rule4RSIdown))]
#drop = final.loc[((rule1dkdown & rule1dRSIdown) & (rule12kdown & rule12RSIdown) & (rule8kdown & rule8RSIdown))]
#drop = final.loc[((rule1dkdown & rule1dRSIdown) & (rule12kdown & rule12RSIdown))]


drop


# In[31]:


##  FULL down options from 1 day till 2 hours  down option and & 
drop = final.loc[((rule1dkdown & rule1dRSIdown) & (rule12kdown & rule12RSIdown) & (rule8kdown & rule8RSIdown) & (rule4kdown & rule4RSIdown))]
drop


# # UP options

# In[32]:


###FULL  up option and | till 8

#drop = final.loc[((rule3dkup | rule3dRSIup) & (rule1dkup | rule1dRSIup) & (rule12kup | rule12RSIup) & (rule8kup | rule8RSIup) & (rule4kup | rule4RSIup) & (rule2kup | rule2RSIup))]
#drop = final.loc[((rule1dkup & rule1dRSIup) & (rule12kup & rule12RSIup) & (rule8kup & rule8RSIup) & (rule4kup & rule4RSIup))]
drop = final.loc[((rule1dkup & rule1dRSIup) & (rule12kup & rule12RSIup) & (rule8kup & rule8RSIup))]

#drop = final.loc[((rule1dkup & rule1dRSIup) | (rule12kup & rule12RSIup) | (rule8kup & rule8RSIup) )]
#drop = final.loc[((rule3dkup | rule3dRSIup) & (rule1dkup | rule1dRSIup) & (rule12kup | rule12RSIup) & (rule8kup | rule8RSIup)) ]
#drop = final.loc[((rule3dkup | rule3dRSIup) & (rule1dkup | rule1dRSIup) & (rule12kup | rule12RSIup) & (rule8kup | rule8RSIup) & (rule4kup | rule4RSIup) ) ]
#drop = drop.loc[(roomr3d & roomr12h & roomr8h & roomr4h)]
#drop = final.loc[((rule3dkup | rule3dRSIup) & (rule1dkup | rule1dRSIup) & (rule12kup | rule12RSIup) & (rule8kup | rule8RSIup) & (rule4kup | rule4RSIup) & (final['STRSIK1d'] < 0.2) ) ]

drop


# In[33]:


# up options but down on 4 

drop = final.loc[((rule3dkup | rule3dRSIup) & (rule1dkup | rule1dRSIup) & (rule12kup | rule12RSIup) & (rule8kup | rule8RSIup) & (dshift4h | downtrend4h)) ]
drop


# In[34]:


### 3 days up option and |

#drop = final.loc[((rule3dkup | rule3dRSIup))]
#drop = final.loc[((rule3dkup | rule3dRSIup) & (final['STRSIK1d'] < 0.2) )]

#drop = final.loc[((rule3dkup & rule3dRSIup) & (rule3dover))]
#drop = final.loc[((rule3dkup & rule3dRSIup) & (rule1dkup))]


#drop


# In[35]:


## 3days under 0.2 // about to go up 
#drop = final.loc[( (final['STRSIK3d'] < 0.2) )]
#drop 


# In[36]:


# 1 day under 0.2 about to go up 
#drop = final.loc[(  (final['STRSIK1d'] < 0.2) )]
#drop


# In[37]:


# 1 and 3 day under 0.2
drop = final.loc[(  (final['STRSIK1d'] < 0.2) & (final['STRSIK3d'] < 0.2) )]
drop


# In[38]:


# 1 and 3 days pointing up and still have room on 3 days < 0.7 

#drop = final.loc[( (final['STRSIK3d'] > final['STRSID3d']) & (final['STRSIK1d'] > final['STRSID1d'])  & (final['STRSIK3d'] < 0.7)  )]
drop = final.loc[( (rule3dkup) & (rule1dkup)  & (final['STRSIK3d'] < 0.8) & (final['STRSIK1d'] < 0.8)  )]

drop


# In[39]:


## 1 day small room up
drop = final.loc[( (rule3dover) & (rule1dover)  & (top1d) & (final['STRSIK1d'] > 0.8)  & (rule1dRSIup) & (rule3dkup) & (rule1dkup) )]
drop


# In[40]:


## 1 day will start go down 
drop = final.loc[( (rule3dover) & (rule1dover)  & (final['STRSIK1d'] > 0.8) & (rule1dRSIdown) )]
drop


# In[41]:


# 1 day started to  go down 
drop = final.loc[( (rule1dkdown) & (rule1dover)  & (final['STRSIK1d'] > 0.8) & (rule1dRSIdown) )]

drop


# In[42]:


# 12 H over and  RSI up
#drop = final.loc[(rule12over & rule12RSIup)]

#drop


# In[43]:


# 12 H down  RSI down
#drop = final.loc[(rule12kdown & rule12RSIdown)]

#drop


# In[44]:


# 12 down  RSI up
#drop = final.loc[(rule12kdown & rule12RSIup)]
#drop = final.loc[(rule12kdown & rule12RSIup)]

#drop


# In[45]:


# 12 down  RSI up under 0.2
drop = final.loc[(rule12kdown & rule12RSIup & (final['STRSIK12h'] < 0.2) )]

#drop = final.loc[(rule12kdown & rule12RSIup)]

drop


# In[46]:


# 12 up  RSI up under 0.2
drop = final.loc[(rule12kup & rule12RSIup & (final['STRSIK12h'] < 0.2) )]
drop = final.loc[((final['STRSIK12h'] < 0.2) )]

#drop = final.loc[(rule12kdown & rule12RSIup)]

drop


# In[47]:


# 8 H down
drop = final.loc[(rule8kdown & rule8RSIdown)]

drop


# In[48]:


# 8 H over RSI up
drop = final.loc[(rule8over & rule8RSIup)]

drop


# In[49]:


# 8 H down RSI up
drop = final.loc[(rule8kdown & rule8RSIup)]

drop


# In[50]:


# 8h under 0.2

drop = final.loc[((final['STRSIK8h']< 0.2) )]
drop


# In[51]:


# 4 H down
drop = final.loc[(rule4kdown & rule4RSIdown)]

drop


# In[73]:


# 4 H reversal start
drop = final.loc[(rule4kup & rule4RSIup )]
drop = final.loc[(rule4kup )]

drop


# In[53]:


# 4 H reversal start
drop = final.loc[(rule4kdown & rule4RSIup &  (final['STRSIK4h']< 0.2))]

drop


# In[54]:


# 4 hour K rising 
drop = final.loc[((rule4under & rule4kup) & (final['STRSIK4h']< 0.2) )]
#drop = final.loc[((rule4under & rule4kup)  )]

#drop = final.loc[(rule12kdown & rule12RSIup)]

drop


# In[55]:


drop = final.loc[((rule4kup & rule4RSIup) )]
drop


# In[56]:


# 4 H down RSI up
drop = final.loc[(rule4kdown & rule4RSIup)]

drop


# In[57]:


# 4H RSI DOWN 
drop = final.loc[(rule4RSIdown)]
drop


# In[58]:


# 4h under 0.2
#drop = final.loc[((rule4under & rule4kup) & (final['STRSIK4h']< 0.2) )]

drop = final.loc[((final['STRSIK4h']< 0.2) )]
drop


# In[59]:


# 2h under 0.2
#drop = final.loc[((rule4under & rule4kup) & (final['STRSIK4h']< 0.2) )]

drop = final.loc[((final['STRSIK2h']< 0.2) )]
drop


# In[60]:


# 2 H down RSI up
drop = final.loc[(rule2kdown & rule2RSIup)]

drop


# In[61]:


# 2 H down RSI down
drop = final.loc[(rule2kdown & rule2RSIdown) ]

drop


# In[62]:


# 2 H up  RSI up
drop = final.loc[(rule2kup & rule2RSIup)]

drop


# # Second part to be cleaned

# In[63]:


# 8 4 2 over and up

drop = final.loc[(rule8over & rule8kup & rule4over & rule4kup & rule2over) ]
#drop = final.loc[(rule4over & rule4kup & rule2over) ]

drop 


# In[64]:


# Split RSI and STCH  tables 
#finalRSI = final.loc[:,['Indicator','RSI2h','RSIM2h','RSI4h','RSIM4h','RSI8h','RSIM8h','RSI12h','RSIM12h','RSI1d','RSIM1d','RSI3d','RSIM3d']]
#finalSTRSIK = final.loc[:,['Indicator','STRSIK2h','STRSIKM2h','STRSIK4h','STRSIKM4h','STRSIK8h','STRSIKM8h','STRSIK12h','STRSIKM12h','STRSIK1d','STRSIKM1d','STRSIK3d','STRSIKM3d','STRSID2h','STRSIDM2h','STRSID4h','STRSIDM4h','STRSID8h','STRSIDM8h','STRSID12h','STRSIDM12h','STRSID1d','STRSIDM1d','STRSID3d','STRSIDM3d']]
finalRSI = final.loc[:,['Indicator','RSI2h','RSIM2h','RSI4h','RSIM4h','RSI8h','RSIM8h','RSI12h','RSIM12h','RSI1d','RSIM1d','RSI3d','RSIM3d']]
finalSTRSIK = final.loc[:,['Indicator','STRSIK2h','STRSIKM2h','STRSIK4h','STRSIKM4h','STRSIK8h','STRSIKM8h','STRSIK12h','STRSIKM12h','STRSIK1d','STRSIKM1d','STRSIK3d','STRSIKM3d','STRSID2h','STRSIDM2h','STRSID4h','STRSIDM4h','STRSID8h','STRSIDM8h','STRSID12h','STRSIDM12h','STRSID1d','STRSIDM1d','STRSID3d','STRSIDM3d']]
#finalST = 
finalSTRSIK


# In[65]:


drop = final.loc[( rule1dover & rule1dkup & rule12under &rule12kdown & rule8under & rule4kdown)]
drop


# In[66]:


## Will go up below 0.2 k 4 and 8 still going up

drop = final.loc[((final['STRSIK8h'] > final['STRSIKM8h']) & (final['STRSIK4h'] > final['STRSIKM4h']) & (final['STRSIK8h'] < 0.2) &  (final['STRSIK8h'] > final['STRSID8h']))]
drop


# In[67]:


## under 0.2 and will go down
## WILL GO DOWN still K > D , K are dropping on 4 hour and 8 hours still above D and under 0.2
drop = final.loc[((final['STRSIK8h'] < final['STRSIKM8h']) & (final['STRSIK4h'] < final['STRSIKM4h']) & (final['STRSIK4h'] < 0.2) &  (final['STRSIK8h'] > final['STRSID8h']))]
drop


# In[68]:


#save = drop[['Indicator','RSI','STRSI']]
#save = save.rename(columns={'STRSI': 'STRSI1','RSI':'RSI1'})
#save
#save.to_csv("strsi1KD.csv")
RSITABLE = drop[['Indicator', 'RSI2h', 'RSIM2h','RSI4h', 'RSIM4h','RSI8h', 'RSIM8h']]
sort1 = RSITABLE.sort_values(by=['RSI2h'],ascending=True,inplace = True)


RSITABLE


# In[69]:


STCHTABLE = drop[['Indicator','STRSIK2h', 'STRSID2h','STRSIK4h', 'STRSID4h','STRSIK8h', 'STRSID8h']]
STCHTABLE


# In[70]:


# High 8 hours going down
STCHTABLEH = STCHTABLE.loc[STCHTABLE['STRSID8h']> 0.8]
STCHTABLEH
#save


# In[71]:


#del final, whitelist ,dflive

