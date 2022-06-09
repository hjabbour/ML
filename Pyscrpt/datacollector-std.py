#!/usr/bin/env python
# coding: utf-8

# In[1]:


## need to add open close for each time frame ,
## support resistance
##  pivots ,https://towardsdatascience.com/pivot-points-calculation-in-python-for-day-trading-659c1e92d323 
## and write it down each 15 minutes to create data set for AI ML bot 
## bot with GAN or CAN 


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
#whitelist
#final = pd.DataFrame(fixentries['Indicator']) 
#final.set_index('Indicator')


# In[4]:


#alllist = 'GALA/USDT, ANT/USDT,IMX/USDT, MATIC/USDT, ZRX/USDT,  ALGO/USDT, BTC/USDT, ETH/USDT, AXS/USDT, SOL/USDT, MATIC/USDT, ADA/USDT, BNB/USDT, FIL/USDT, XRP/USDT, CVC/USDT, FTM/USDT, ICP/USDT, DOGE/USDT, XEC/USDT, IOTA/USDT, AVAX/USDT, DOT/USDT, LTC/USDT, EOS/USDT, IOST/USDT, STMX/USDT, FTT/USDT, TWT/USDT, ALICE/USDT, ATA/USDT, ETC/USDT, VET/USDT, SAND/USDT, LINK/USDT, THETA/USDT, SHIB/USDT, CHZ/USDT, LUNA/USDT, IOTX/USDT, TRX/USDT, TLM/USDT, BCH/USDT, REEF/USDT, CAKE/USDT, NEO/USDT, SLP/USDT, RVN/USDT, ATOM/USDT, SUSHI/USDT, ARDR/USDT, OMG/USDT, GRT/USDT, 1INCH/USDT, AAVE/USDT, C98/USDT, UNI/USDT, HBAR/USDT, CRV/USDT, DENT/USDT, MBOX/USDT, BTT/USDT, SRM/USDT, YFI/USDT, HOT/USDT, XLM/USDT, FIS/USDT, REQ/USDT, SXP/USDT, KSM/USDT, QNT/USDT, NEAR/USDT, COMP/USDT, QTUM/USDT, CHR/USDT, SC/USDT, XVS/USDT, ONT/USDT, COTI/USDT, NKN/USDT, RUNE/USDT, ANKR/USDT, MANA/USDT, CTXC/USDT, AR/USDT, RAY/USDT, BAKE/USDT, ARPA/USDT, XTZ/USDT, ZIL/USDT, ONG/USDT, ROSE/USDT, PNT/USDT, ALGO/USDT, MINA/USDT, WIN/USDT, AUDIO/USDT, STORJ/USDT, EPS/USDT, MDX/USDT, ZEC/USDT, ENJ/USDT, HIVE/USDT, TFUEL/USDT, WRX/USDT, SNX/USDT, SKL/USDT, LINA/USDT, DODO/USDT, UNFI/USDT, XEM/USDT, EGLD/USDT, ICX/USDT, WAVES/USDT, ONE/USDT, DASH/USDT, KAVA/USDT, SUPER/USDT, TKO/USDT, OGN/USDT, XMR/USDT, OCEAN/USDT, CELO/USDT, BAT/USDT, FUN/USDT, LRC/USDT, ALPHA/USDT, MASK/USDT, TRU/USDT, DNT/USDT, LIT/USDT, FET/USDT, GTC/USDT, CELR/USDT, DEXE/USDT, ZEN/USDT, WAXP/USDT, CTSI/USDT, HNT/USDT, YFII/USDT, BEL/USDT, ZRX/USDT, DEGO/USDT, RSR/USDT, COS/USDT, FLM/USDT, BAL/USDT, MFT/USDT, RLC/USDT, KNC/USDT, REN/USDT, INJ/USDT, FLOW/USDT, SFP/USDT, BAND/USDT, CTK/USDT, SUN/USDT, AKRO/USDT, XVG/USDT, NANO/USDT, STRAX/USDT, PUNDIX/USDT, TVK/USDT, MKR/USDT, CLV/USDT, TRB/USDT, ETHUP/USDT, KEEP/USDT, TCT/USDT, POND/USDT, LTO/USDT, ANT/USDT, TOMO/USDT, ACM/USDT, FIO/USDT, VITE/USDT, MBL/USDT, JST/USDT, MTL/USDT, UTK/USDT, BLZ/USDT, ORN/USDT, STPT/USDT, CFX/USDT, OM/USDT, PERP/USDT, MIR/USDT, STX/USDT, ERN/USDT, TUSD/USDT, PSG/USDT, NULS/USDT, OXT/USDT, REP/USDT, BNBUP/USDT, DATA/USDT, WAN/USDT, BURGER/USDT, RAMP/USDT, ATM/USDT, DGB/USDT, DIA/USDT, BEAM/USDT, UMA/USDT, ETHDOWN/USDT, CKB/USDT, TRIBE/USDT, FARM/USDT, BTCUP/USDT, BZRX/USDT, ALPACA/USDT, DOTUP/USDT, BTS/USDT, FOR/USDT, ADAUP/USDT, HARD/USDT, LSK/USDT, POLS/USDT, FORTH/USDT, TORN/USDT, VTHO/USDT, BTG/USDT, QUICK/USDT, FILUP/USDT, PAX/USDT, MITH/USDT, SUSHIUP/USDT, BTCDOWN/USDT, XRPDOWN/USDT, BNBDOWN/USDT, DUSK/USDT, XRPUP/USDT, WTC/USDT, DOCK/USDT, AUTO/USDT, BTCST/USDT, KLAY/USDT, WNXM/USDT, KEY/USDT, FILDOWN/USDT, NU/USDT, IRIS/USDT, KMD/USDT, TROY/USDT, WING/USDT, AION/USDT, PHA/USDT, SUSHIDOWN/USDT, EOSUP/USDT, LTCUP/USDT, GHST/USDT, PAXG/USDT, BNT/USDT, AVA/USDT, BADGER/USDT, PERL/USDT, COCOS/USDT, GXS/USDT, MDT/USDT, 1INCHUP/USDT, GTO/USDT, LINKUP/USDT, AAVEUP/USDT, FIRO/USDT, JUV/USDT, MLN/USDT, UNIUP/USDT, LPT/USDT, NMR/USDT, AAVEDOWN/USDT, 1INCHDOWN/USDT, ASR/USDT, OG/USDT, BAR/USDT, DREP/USDT, ADADOWN/USDT, BOND/USDT, DCR/USDT, DOTDOWN/USDT, GNO/USDT, NBS/USDT, SXPUP/USDT, EOSDOWN/USDT, YFIUP/USDT, RIF/USDT, LTCDOWN/USDT, XLMUP/USDT, YFIDOWN/USDT, LINKDOWN/USDT, TRXUP/USDT, BCHUP/USDT, XTZUP/USDT, SXPDOWN/USDT, UNIDOWN/USDT, XTZDOWN/USDT, BCHDOWN/USDT, TRXDOWN/USDT, XLMDOWN/USDT, SUSD/USDT'
#whitelist = 'LINK/USDT, DOT/USDT'
#whitelist =str(fixentries['Indicator'])

startdate = '2020-01-01'
enddate = pd.to_datetime('today')
intervals = ['5m','15m','30m','1h','2h','4h','8h','12h','1d','3d','1w']
#intervals = ['8h']


# In[5]:


#print (enddate)


# In[6]:


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

## MACD 
def MMACD(series,fast=5,slow=35,sig=5):
    ## defaults original f 12, s 26, s9
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1-exp2
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd,signal
## Bottom price 
def FBOTTOMP (series,size):
    bottomp=series.rolling(size).min()
    return bottomp
## top price 
def FTOPP (series,size):
    topp=series.rolling(size).max()
    return topp



# In[7]:


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
final.set_index(['Indicator'],inplace=True)
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
                dflive['SMA21'] = SMA(dflive['c'],21)
                dflive['SMA55'] = SMA(dflive['c'],55)
                dflive['SMA100'] = SMA(dflive['c'],100)
                dflive['SMA200'] = SMA(dflive['c'],200)
                dflive['MACD'] = MMACD(dflive['c'])[0]
                dflive['MACDSIGNAL'] = MMACD(dflive['c'])[1]
                dflive['BTTOMP'] = FBOTTOMP(dflive['l'],10)
                dflive['TPP'] = FTOPP(dflive['h'],10)


                
                



                #print(dflive)


                lastrow = dflive.iloc[-1]
                beforelast = dflive.iloc[-2]
                row5minus = dflive.iloc[-10]
                O = float(lastrow.o)
                C = float(lastrow.c)
                H = float(lastrow.h)
                L = float(lastrow.l)
                V = float(lastrow.v)
                OM = float(beforelast.o)
                CM = float(beforelast.c)
                HM = float(beforelast.h)
                LM = float(beforelast.l)
                VM = float(beforelast.v)
                RSI = float(lastrow.RSI)
                RSIM = float(beforelast.RSI)
                STRSI = float(lastrow.STRSI)
                STRSIK = float(lastrow.STRSIK)
                STRSIKM = float(beforelast.STRSIK)
                STRSID = float(lastrow.STRSID)
                STRSIDM = float(beforelast.STRSID)
                SMA21 = float(lastrow.SMA21)
                SMA21M = float(beforelast.SMA21)
                SMA55 = float(lastrow.SMA55)
                SMA55M = float(beforelast.SMA55)
                SMA100 = float(lastrow.SMA100)
                SMA100M = float(beforelast.SMA100)
                SMA200 = float(lastrow.SMA200)
                SMA200M = float(beforelast.SMA200)
                MACD = float(lastrow.MACD)
                MACDSIGNAL = float(lastrow.MACDSIGNAL)
                MACDM = float(beforelast.MACD)
                MACDSIGNALM = float(beforelast.MACDSIGNAL)
                BTTOMP = lastrow.BTTOMP
                TPP = lastrow.TPP
                BTTOMPM = row5minus.BTTOMP
                TPPM = row5minus.TPP


                tuple =(symbol,O,C,H,L,V,OM,CM,HM,LM,VM,RSI,RSIM,STRSI,STRSIK,STRSIKM,STRSID,STRSIDM,SMA21,SMA21M,SMA55,SMA55M,SMA100,SMA100M,SMA200,SMA200M,MACD,MACDSIGNAL,MACDM,MACDSIGNALM,BTTOMP,TPP,BTTOMPM,TPPM)
                table.append(tuple)
                #print(table)
                del dflive
            else:
                pass
            #print("name is: ",array[0])


    table[0]
    AllTime = pd.DataFrame(table,columns=['Indicator','O','C','H','L','V','OM','CM','HM','LM','VM','RSI','RSIM','STRSI','STRSIK','STRSIKM','STRSID','STRSIDM','SMA21','SMA21M','SMA55','SMA55M','SMA100','SMA100M','SMA200','SMA200M','MACD','MACDSIGNAL','MACDM','MACDSIGNALM','BOTTOMP','TOPP','BOTTOMPM','TOPPM'])


    #AllTime.Current=AllTime.Current.astype(float)

    AllTime.dtypes


    #AllTime.sort_values(by='ATH',ascending=False)
    #AllTime.to_csv("alltimeHL.csv", mode="w")
    #data.sort_values("Name", axis = 0, ascending = True,inplace = True, na_position ='last')

    ## Display the Data 

    test=AllTime.rename(columns={'O':('O'+interval),'OM':('OM'+interval),'C':('C'+interval),'CM':('CM'+interval),'H':('H'+interval),'HM':('HM'+interval),'L':('L'+interval),'LM':('LM'+interval),'V':('V'+interval),'VM':('VM'+interval),'RSI':('RSI'+interval),                                 'RSIM':('RSIM'+interval),'STRSI':('STRSI'+interval),'STRSIK':('STRSIK'+interval),'STRSIKM':('STRSIKM'+interval),                                 'STRSID':('STRSID'+interval),'STRSIDM':('STRSIDM'+interval),'SMA21':('SMA21'+interval),'SMA21M':('SMA21M'+interval),                                 'SMA55':('SMA55'+interval),'SMA55M':('SMA55M'+interval),'SMA100':('SMA100'+interval),'SMA100M':('SMA100M'+interval),                                 'SMA200':('SMA200'+interval),'SMA200M':('SMA200M'+interval),'MACD':('MACD'+interval),'MACDSIGNAL':('MACDSIGNAL'+interval),'MACDM':('MACDM'+interval),'MACDSIGNALM':('MACDSIGNALM'+interval),'BOTTOMP':('BOTTOMP'+interval),'TOPP':('TOPP'+interval),'BOTTOMPM':('BOTTOMPM'+interval),'TOPPM':('TOPPM'+interval)})
    #test.set_index(['Indicator'],inplace=True)
    final=final.merge(test,how='left',on=['Indicator'])
    final=final.drop_duplicates(subset='Indicator',keep='last')
    #final.set_index(['Indicator'],inplace=True)

    del test
    ## add the current date column
            
final['Time'] = pd.to_datetime('today')

#final
#final.reset_index(inplace=True)


# In[8]:


#final.Indicator


# In[9]:


import os.path

file_exists = os.path.exists("MLDATATOP"+str(date.today())+".csv")



#final=final.drop_duplicates(subset='Indicator',keep='last')
#final= final.drop(columns=['Current_x','Current_y'])
#final['Time'] = pd.to_datetime('today')
#final.to_csv("MLDATATOP.csv")
#final.to_csv("MLSTRSIMACDBTTOP.csv")
if file_exists :
    final.to_csv("MLDATATOP"+str(date.today())+".csv", mode='a', index=True, header=False)
else :
    final.to_csv("MLDATATOP"+str(date.today())+".csv")
    
#final.to_csv("MLDATATOP"+str(date.today())+".csv")
## adding to two files the first file i thought there is a column missmatch
#final.to_csv('MLSTRSIMACDBTTOP.csv', mode='a', index=True, header=False)
final.to_csv('MLDATATOP.csv', mode='a', index=True, header=False)
#final.reset_index(inplace=True)

final.head()


# In[10]:


final.tail()


# In[11]:


HIHI15m = (final['TOPP15m'] > final['TOPPM15m'])
HIHI1h = (final['TOPP1h'] > final['TOPPM1h'])
HIHI2h = (final['TOPP2h'] > final['TOPPM2h'])
HIHI30m = (final['TOPP30m'] > final['TOPPM30m'])
HIHI4h = (final['TOPP4h'] > final['TOPPM4h'])
HIHI5m = (final['TOPP5m'] > final['TOPPM5m'])
HILO15m = (final['BOTTOMP15m'] > final['BOTTOMPM15m'])
HILO1h = (final['BOTTOMP1h'] > final['BOTTOMPM1h'])
HILO2h = (final['BOTTOMP2h'] > final['BOTTOMPM2h'])
HILO30m = (final['BOTTOMP30m'] > final['BOTTOMPM30m'])
HILO4h = (final['BOTTOMP4h'] > final['BOTTOMPM4h'])
HILO5m = (final['BOTTOMP5m'] > final['BOTTOMPM5m'])
LOHI15m = (final['TOPP15m'] < final['TOPPM15m'])
LOHI1h = (final['TOPP1h'] < final['TOPPM1h'])
LOHI2h = (final['TOPP2h'] < final['TOPPM2h'])
LOHI30m = (final['TOPP30m'] < final['TOPPM30m'])
LOHI4h = (final['TOPP4h'] < final['TOPPM4h'])
LOHI5m = (final['TOPP5m'] < final['TOPPM5m'])
LOLO15m = (final['BOTTOMP15m'] < final['BOTTOMPM15m'])
LOLO1h = (final['BOTTOMP1h'] < final['BOTTOMPM1h'])
LOLO2h = (final['BOTTOMP2h'] < final['BOTTOMPM2h'])
LOLO30m = (final['BOTTOMP30m'] < final['BOTTOMPM30m'])
LOLO4h = (final['BOTTOMP4h'] < final['BOTTOMPM4h'])
LOLO5m = (final['BOTTOMP5m'] < final['BOTTOMPM5m'])




MACDCROSS12h =  (final['MACD12h'] > final['MACDSIGNAL12h'])
MACDCROSS1d =  (final['MACD1d'] > final['MACDSIGNAL1d'])
MACDCROSS1w =  (final['MACD1w'] > final['MACDSIGNAL1w'])
MACDCROSS3d =  (final['MACD3d'] > final['MACDSIGNAL3d'])
MACDCROSS4h =  (final['MACD4h'] > final['MACDSIGNAL4h'])
MACDCROSS8h =  (final['MACD8h'] > final['MACDSIGNAL8h'])

MACDD12h = (final['MACD12h'] < final['MACDM12h'])
MACDD1d = (final['MACD1d'] < final['MACDM1d'])
MACDD1w = (final['MACD1w'] < final['MACDM1w'])
MACDD3d = (final['MACD3d'] < final['MACDM3d'])
MACDD4h = (final['MACD4h'] < final['MACDM4h'])
MACDD8h = (final['MACD8h'] < final['MACDM8h'])

MACDGAPD12h = ((final['MACD12h'] - final['MACDSIGNAL12h']) > (final['MACDM12h'] - final['MACDSIGNALM12h']) )
MACDGAPD1d = ((final['MACD1d'] - final['MACDSIGNAL1d']) > (final['MACDM1d'] - final['MACDSIGNALM1d']) )
MACDGAPD1w = ((final['MACD1w'] - final['MACDSIGNAL1w']) > (final['MACDM1w'] - final['MACDSIGNALM1w']) )
MACDGAPD3d = ((final['MACD3d'] - final['MACDSIGNAL3d']) > (final['MACDM3d'] - final['MACDSIGNALM3d']) )
MACDGAPD4h = ((final['MACD4h'] - final['MACDSIGNAL4h']) > (final['MACDM4h'] - final['MACDSIGNALM4h']) )
MACDGAPD8h = ((final['MACD8h'] - final['MACDSIGNAL8h']) > (final['MACDM8h'] - final['MACDSIGNALM8h']) )
MACDGAPU12h = ((final['MACD12h'] - final['MACDSIGNAL12h']) < (final['MACDM12h'] - final['MACDSIGNALM12h']) )
MACDGAPU1d = ((final['MACD1d'] - final['MACDSIGNAL1d']) < (final['MACDM1d'] - final['MACDSIGNALM1d']) )
MACDGAPU1w = ((final['MACD1w'] - final['MACDSIGNAL1w']) < (final['MACDM1w'] - final['MACDSIGNALM1w']) )
MACDGAPU3d = ((final['MACD3d'] - final['MACDSIGNAL3d']) < (final['MACDM3d'] - final['MACDSIGNALM3d']) )
MACDGAPU4h = ((final['MACD4h'] - final['MACDSIGNAL4h']) < (final['MACDM4h'] - final['MACDSIGNALM4h']) )
MACDGAPU8h = ((final['MACD8h'] - final['MACDSIGNAL8h']) < (final['MACDM8h'] - final['MACDSIGNALM8h']) )

MACDU12h = (final['MACD12h'] > final['MACDM12h'])
MACDU1d = (final['MACD1d'] > final['MACDM1d'])
MACDU1w = (final['MACD1w'] > final['MACDM1w'])
MACDU3d = (final['MACD3d'] > final['MACDM3d'])
MACDU4h = (final['MACD4h'] > final['MACDM4h'])
MACDU8h = (final['MACD8h'] > final['MACDM8h'])

bottom12h = (final['STRSIK12h']< 0.2)
bottom1d = (final['STRSIK1d']< 0.2)
bottom1h = (final['STRSIK1h']< 0.2)
bottom1w = (final['STRSIK1w']< 0.2)
bottom2h = (final['STRSIK2h']< 0.2)
bottom3d = (final['STRSIK3d']< 0.2)
bottom4h = (final['STRSIK4h']< 0.2)
bottom8h = (final['STRSIK8h']< 0.2)
bottom30m = (final['STRSIK30m'] < 0.2)
bottom15m = (final['STRSIK15m'] < 0.2)
bottom5m = (final['STRSIK5m'] < 0.2)

bottomr12h = (final['RSI12h'] < 30)
bottomr1d = (final['RSI1d'] < 30)
bottomr1h = (final['RSI1h'] < 30)
bottomr1w = (final['RSI1w'] < 30)
bottomr2h = (final['RSI2h'] < 30)
bottomr3d = (final['RSI3d'] < 30)
bottomr4h = (final['RSI4h'] < 30)
bottomr8h = (final['RSI8h'] < 30)
bottomr30m = (final['RSI30m'] < 30)
bottomr15m = (final['RSI15m'] < 30)
bottomr5m = (final['RSI5m'] < 30)






room12h = (final['STRSIK12h'] < 0.8)
room1d = (final['STRSIK1d'] < 0.8)
room1h = (final['STRSIK1h'] < 0.8)
room1w = (final['STRSIK1w'] < 0.8)
room2h = (final['STRSIK2h'] < 0.8)
room3d = (final['STRSIK3d'] < 0.8)
room4h = (final['STRSIK4h'] < 0.8)
room8h = (final['STRSIK8h'] < 0.8)
room30m = (final['STRSIK30m'] < 0.8)
room15m = (final['STRSIK15m'] < 0.8)
room5m = (final['STRSIK5m'] < 0.8)


roomr12h = (final['RSI12h'] < 70)
roomr1d = (final['RSI1d'] < 70)
roomr1h = (final['RSI1h'] < 70)
roomr1w = (final['RSI1w'] < 70)
roomr2h = (final['RSI2h'] < 70)
roomr3d = (final['RSI3d'] < 70)
roomr4h = (final['RSI4h'] < 70)
roomr8h = (final['RSI8h'] < 70)
roomr30m = (final['RSI30m'] < 70)
roomr15m = (final['RSI15m'] < 70)
roomr5m = (final['RSI5m'] < 70)


rule12hRSIdown = (final['RSI12h'] < final['RSIM12h'])
rule12hRSIup = (final['RSI12h'] > final['RSIM12h'])
rule12hkdown = (final['STRSIK12h'] < final['STRSIKM12h'])
rule12hkup = (final['STRSIK12h'] > final['STRSIKM12h'])
rule12hover = (final['STRSIK12h'] > final['STRSID12h'])
rule12hunder = (final['STRSIK12h'] < final['STRSID12h'])
rule1hRSIdown = (final['RSI1h'] < final['RSIM1h'])
rule1hRSIup = (final['RSI1h'] > final['RSIM1h'])
rule1dRSIdown = (final['RSI1d'] < final['RSIM1d'])
rule1dRSIup = (final['RSI1d'] > final['RSIM1d'])
rule1dkdown = (final['STRSIK1d']< final['STRSIKM1d'])
rule1dkup = (final['STRSIK1d'] > final['STRSIKM1d'])
rule1dover = (final['STRSIK1d'] > final['STRSID1d'])
rule1dunder = (final['STRSIK1d']< final['STRSID1d'])
rule1hkdown = (final['STRSIK1h']< final['STRSIKM1h'])
rule1hkup = (final['STRSIK1h'] > final['STRSIKM1h'])
rule1hover = (final['STRSIK1h'] > final['STRSID1h'])
rule1hunder = (final['STRSIK1h']< final['STRSID1h'])
rule1wRSIdown = (final['RSI1w'] < final['RSIM1w'])
rule1wRSIup = (final['RSI1w'] > final['RSIM1w'])
rule1wkdown = (final['STRSIK1w']< final['STRSIKM1w'])
rule1wkup = (final['STRSIK1w'] > final['STRSIKM1w'])
rule1wover = (final['STRSIK1w'] > final['STRSID1w'])
rule1wunder = (final['STRSIK1w']< final['STRSID1w'])
rule2hRSIdown = (final['RSI2h'] < final['RSIM2h'])
rule2hRSIup = (final['RSI2h'] > final['RSIM2h'])
rule2hkdown = (final['STRSIK2h']< final['STRSIKM2h'])
rule2hkup = (final['STRSIK2h'] > final['STRSIKM2h'])
rule2hover = (final['STRSIK2h'] > final['STRSID2h'])
rule2hunder = (final['STRSIK2h']< final['STRSID2h'])
rule3dRSIdown = (final['RSI3d'] < final['RSIM3d'])
rule3dRSIup = (final['RSI3d'] > final['RSIM3d'])
rule3dkdown = (final['STRSIK3d']< final['STRSIKM3d'])
rule3dkup = (final['STRSIK3d'] > final['STRSIKM3d'])
rule3dover = (final['STRSIK3d'] > final['STRSID3d'])
rule3dunder = (final['STRSIK3d']< final['STRSID3d'])
rule4hRSIdown = (final['RSI4h'] < final['RSIM4h'])
rule4hRSIup = (final['RSI4h'] > final['RSIM4h'])
rule4hkdown = (final['STRSIK4h']< final['STRSIKM4h'])
rule4hkup = (final['STRSIK4h'] > final['STRSIKM4h'])
rule4hover = (final['STRSIK4h'] > final['STRSID4h'])
rule4hunder = (final['STRSIK4h']< final['STRSID4h'])

rule8hRSIdown = (final['RSI8h'] < final['RSIM8h'])
rule8hRSIup = (final['RSI8h'] > final['RSIM8h'])
rule8hkdown = (final['STRSIK8h']< final['STRSIKM8h'])
rule8hkup = (final['STRSIK8h'] > final['STRSIKM8h'])
rule8hover = (final['STRSIK8h'] > final['STRSID8h'])
rule8hunder = (final['STRSIK8h']< final['STRSID8h'])

rule30mRSIdown = (final['RSI30m'] < final['RSIM30m'])
rule30mRSIup = (final['RSI30m'] > final['RSIM30m'])
rule30mkdown = (final['STRSIK30m']< final['STRSIKM30m'])
rule30mkup = (final['STRSIK30m'] > final['STRSIKM30m'])
rule30mover = (final['STRSIK30m'] > final['STRSID30m'])
rule30munder = (final['STRSIK30m']< final['STRSID30m'])

rule15mRSIdown = (final['RSI15m'] < final['RSIM15m'])
rule15mRSIup = (final['RSI15m'] > final['RSIM15m'])
rule15mkdown = (final['STRSIK15m']< final['STRSIKM15m'])
rule15mkup = (final['STRSIK15m'] > final['STRSIKM15m'])
rule15mover = (final['STRSIK15m'] > final['STRSID15m'])
rule15munder = (final['STRSIK15m']< final['STRSID15m'])

rule5mRSIdown = (final['RSI5m'] < final['RSIM5m'])
rule5mRSIup = (final['RSI5m'] > final['RSIM5m'])
rule5mkdown = (final['STRSIK5m']< final['STRSIKM5m'])
rule5mkup = (final['STRSIK5m'] > final['STRSIKM5m'])
rule5mover = (final['STRSIK5m'] > final['STRSID5m'])
rule5munder = (final['STRSIK5m']< final['STRSID5m'])

shift12h = (rule12hunder & (rule12hkup | rule12hRSIup))
shift1d = (rule1dunder & (rule1dkup | rule1dRSIup))
shift1h = (rule1hunder & (rule1hkup | rule1hRSIup))
shift1w = (rule1wunder & (rule1wkup | rule1wRSIup))
shift2h = (rule2hunder & (rule2hkup | rule2hRSIup))
shift3d = (rule3dunder & (rule3dkup | rule3dRSIup))
shift4h = (rule4hunder & (rule4hkup | rule4hRSIup))
shift8h = (rule8hunder & (rule8hkup | rule8hRSIup))
shift30m = (rule30munder & (rule30mkup | rule30mRSIup))
shift15m = (rule15munder & (rule15mkup | rule15mRSIup))
shift5m = (rule5munder & (rule5mkup | rule5mRSIup))



stk12hcross = (final['STRSIKM12h'] < final['STRSIDM12h']) & (final['STRSIK12h'] > final['STRSID12h'])
stk15mcross = (final['STRSIKM15m'] < final['STRSIDM15m']) & (final['STRSIK15m'] > final['STRSID15m'])
stk1hcross = (final['STRSIKM1h'] < final['STRSIDM1h']) & (final['STRSIK1h'] > final['STRSID1h'])
stk1dcross = (final['STRSIKM1d'] < final['STRSIDM1d']) & (final['STRSIK1d'] > final['STRSID1d'])
stk1wcross = (final['STRSIKM1w'] < final['STRSIDM1w']) & (final['STRSIK1w'] > final['STRSID1w'])
stk2hcross = (final['STRSIKM2h'] < final['STRSIDM2h']) & (final['STRSIK2h'] > final['STRSID2h'])
stk30mcross = (final['STRSIKM30m'] < final['STRSIDM30m']) & (final['STRSIK30m'] > final['STRSID30m'])
stk3dcross = (final['STRSIKM3d'] < final['STRSIDM3d']) & (final['STRSIK3d'] > final['STRSID3d'])
stk4hcross = (final['STRSIKM4h'] < final['STRSIDM4h']) & (final['STRSIK4h'] > final['STRSID4h'])
stk5mcross = (final['STRSIKM5m'] < final['STRSIDM5m']) & (final['STRSIK5m'] > final['STRSID5m'])
stk8hcross = (final['STRSIKM8h'] < final['STRSIDM8h']) & (final['STRSIK8h'] > final['STRSID8h'])

top12h = (final['STRSIK12h'] > 0.8)
top1d = (final['STRSIK1d'] > 0.8)
top1h = (final['STRSIK1h'] > 0.8)
top1w = (final['STRSIK1w'] > 0.8)
top2h = (final['STRSIK2h'] > 0.8)
top3d = (final['STRSIK3d'] > 0.8)
top4h = (final['STRSIK4h'] > 0.8)
top8h = (final['STRSIK8h'] > 0.8)
top30m = (final['STRSIK30m'] > 0.8)
top15m = (final['STRSIK15m'] > 0.8)
top5m = (final['STRSIK5m'] > 0.8)



topr12h = (final['RSI12h'] > 70)
topr1d = (final['RSI1d'] > 70)
topr1h = (final['RSI1h'] > 70)
topr1w = (final['RSI1w'] > 70)
topr2h = (final['RSI2h'] > 70)
topr3d = (final['RSI3d'] > 70)
topr4h = (final['RSI4h'] > 70)
topr8h = (final['RSI8h'] > 70)
topr30m = (final['RSI30m'] > 70)
topr15m = (final['RSI15m'] > 70)
topr5m = (final['RSI5m'] > 70)




uptrend12h = (rule12hover & (rule12hkup & rule12hRSIup))
uptrend1d = (rule1dover & (rule1dkup & rule1dRSIup))
uptrend1h = (rule1hover & (rule1hkup & rule1hRSIup))
uptrend1w = (rule1wover & (rule1wkup & rule1wRSIup))
uptrend2h = (rule2hover & (rule2hkup & rule2hRSIup))
uptrend3d = (rule3dover & (rule3dkup & rule3dRSIup))
uptrend4h = (rule4hover & (rule4hkup & rule4hRSIup))
uptrend8h = (rule8hover & (rule8hkup & rule8hRSIup))
uptrend30m = (rule30mover & (rule30mkup & rule30mRSIup))
uptrend15m = (rule15mover & (rule15mkup & rule15mRSIup))
uptrend5m = (rule5mover & (rule5mkup & rule5mRSIup))




downtrend12h = (rule12hunder & (rule12hkdown & rule12hRSIdown))
downtrend1d = (rule1dunder & (rule1dkdown & rule1dRSIdown))
downtrend1h = (rule1hunder & (rule1hkdown & rule1hRSIdown))
downtrend1w = (rule1wunder & (rule1wkdown & rule1wRSIdown))
downtrend2h = (rule2hunder & (rule2hkdown & rule2hRSIdown))
downtrend3d = (rule3dunder & (rule3dkdown & rule3dRSIdown))
downtrend4h = (rule4hunder & (rule4hkdown & rule4hRSIdown))
downtrend8h = (rule8hunder & (rule8hkdown & rule8hRSIdown))
downtrend30m = (rule30munder & (rule30mkdown & rule30mRSIdown))
downtrend15m = (rule15munder & (rule15mkdown & rule15mRSIdown))
downtrend5m = (rule5munder & (rule5mkdown & rule5mRSIdown))


dshift12h = (rule12hover & (rule12hkdown | rule12hRSIdown))
dshift1d = (rule1dover & (rule1dkdown | rule1dRSIdown))
dshift1h = (rule1hover & (rule1hkdown | rule1hRSIdown))
dshift1w = (rule1wover & (rule1wkdown | rule1wRSIdown))
dshift2h = (rule2hover & (rule2hkdown | rule2hRSIdown))
dshift3d = (rule3dover & (rule3dkdown | rule3dRSIdown))
dshift4h = (rule4hover & (rule4hkdown | rule4hRSIdown))
dshift8h = (rule8hover & (rule8hkdown | rule8hRSIdown))
dshift30m = (rule30mover & (rule30mkdown | rule30mRSIdown))
dshift15m = (rule15mover & (rule15mkdown | rule15mRSIdown))
dshift5m = (rule5mover & (rule5mkdown | rule5mRSIdown))


# In[12]:


wshift = final.loc[shift1w]
wshift


# # Summary  draft rules
# 

# In[43]:


#final.reset_index(inplace=True)

#final.set_index(['Indicator'],inplace=True)

# drop = final.loc[bottom4h & (shift4h | rule4over)]
#drop = final.loc[(bottom4h | (uptrend4h | shift4h)) & (uptrend8h | shift8h) ]
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

#drop = final.loc[( (bottom4h) & (uptrend8h | shift8h) & (uptrend1d | shift1d) & (uptrend12h | shift12h)) ]
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

#drop



    

#bottomtable['bottoms'] = bottomtable.apply(get_key(listb,bottomtable.loc('Indicator'),axis=1)

#bottomtable
#reversetable
#bottomstable = pd.DataFrame(reversetable,columns=['Indicator','Bottoms'])
#print (bottomstable)
#listb
#listd = pd.DataFrame(listb,columns=['Bottoms','Indicator'])
#print (listb)
#print(reversetable)
#listd.head()
#test = final.loc[(MACDGAP3d)]
test = final.loc[(MACDCROSS3d)]
test = final.loc[(MACDCROSS1d)]
test = final.loc[(MACDGAPU3d)]
test = final.loc[(MACDGAPD3d)& rule3dover & rule3dRSIup]
#macd3d = final.loc[(MACDGAPU3d)& rule3dRSIup]
macd1d = final.loc[(MACDGAPD1d)& rule1dRSIup]
macd3du = final.loc[(MACDGAPU3d)]
macd3dd = final.loc[(MACDGAPD3d)]
macd1du = final.loc[(MACDGAPU1d)]
macd1dd = final.loc[(MACDGAPD1d)]
macd1wu = final.loc[(MACDGAPU1w)]
macd1wd = final.loc[(MACDGAPD1w)]

macd3d1d = final.loc[(MACDGAPD1d &  (MACDGAPD3d)  & rule3dRSIup & rule1dRSIup )]
macd1dcross = final.loc[MACDCROSS1d]
macd12hcross = final.loc[MACDCROSS12h]
macd8hcross = final.loc[MACDCROSS8h]
macd4hcross = final.loc[MACDCROSS4h]
macd3dcross = final.loc[MACDCROSS3d]
stk3d = final.loc[stk3dcross]
stk1w = final.loc[stk1wcross]
macd3dd
#macd12hcross
#macd12hcross
#macd4hcross
#macd3d1d
#macd1d
#pmacdu = macd3du[['Indicator','MACD3d','MACDSIGNAL3d','MACDM3d','MACDSIGNALM3d']]
#pmacdu['diff']=abs(macd3du['MACD3d'])-abs(macd3du['MACDSIGNAL3d'])
#pmacdu['diffm']=abs(macd3du['MACDM3d'])-abs(macd3du['MACDSIGNALM3d'])

#pmacdd = macd3dd[['Indicator','MACD3d','MACDSIGNAL3d','MACDM3d','MACDSIGNALM3d']]
#pmacdd['diff']=abs(macd3dd['MACD3d'])-abs(macd3dd['MACDSIGNAL3d'])
#pmacdd['diffm']=abs(macd3dd['MACDM3d'])-abs(macd3dd['MACDSIGNALM3d'])
#final.set_index(['Indicator'],inplace=True)

#final.reset_index(inplace=True)
test1h = final[['Indicator','BOTTOMP1h','BOTTOMPM1h','TOPP1h','TOPPM1h']]
#test1h = final[['BOTTOMP1h','BOTTOMPM1h','TOPP1h','TOPPM1h']]

test15 = final[['Indicator','BOTTOMP15m','BOTTOMPM15m','TOPP15m','TOPPM15m']]
#test15.set_index(['Indicator'],inplace=True)
test5 = final[['Indicator','BOTTOMP5m','BOTTOMPM5m','TOPP5m','TOPPM5m']]
test30 = final[['Indicator','BOTTOMP30m','BOTTOMPM30m','TOPP30m','TOPPM30m']]
#test30.set_index(['Indicator'],inplace=True)

test2h = final[['Indicator','BOTTOMP2h','BOTTOMPM2h','TOPP2h','TOPPM2h']]

test4h = final[['Indicator','BOTTOMP4h','BOTTOMPM4h','TOPP4h','TOPPM4h']]

#final.set_index(['Indicator'],inplace=True)

higherhi = test15.loc[HIHI15m & HILO15m]
#higherhi = test30.loc[HIHI30m & HILO30m]
#higherhi = test1h.loc[HIHI1h & HILO1h]
#higherhi = test2h.loc[HIHI2h & HILO2h]

#higherhi = test4h.loc[HIHI4h & HILO4h]

lowhi = test15.loc[LOHI15m & LOLO15m]
lowhi = test5.loc[LOHI5m & LOLO5m]
#lowhi = test1h.loc[LOHI1h & LOLO1h]
lowhi = test30.loc[LOHI30m & LOLO30m]
lowhi = test2h.loc[LOHI2h & LOLO2h]
lowhi = test4h.loc[LOHI4h & LOLO4h]

higherhi5 = test5.loc[HIHI5m & HILO5m]
higherhi2 = test2h.loc[HIHI2h & HILO2h]
higherhi4 = test4h.loc[HIHI4h & HILO4h]


#higherhi = test15.loc[HIHI15m ]
#higherhi = test.loc[HIHI1h]
#higherhi = test5.loc[HIHI5m ]
#higherlo = test5.loc[HILO5m ]
#higherlo = test15.loc[HILO15m ]


higherhi
stk3d
macd3dd
macd1wd
#macd3dd
#macd1dd
#macd1du
#lowhi
hihi5macd1w=macd1wd.loc[HIHI5m & HILO5m & rule30mRSIdown &rule1hRSIup & rule2hRSIup & room2h]
hihi5macd1w
higherhi5


# In[ ]:





# # Functions
# 

# In[14]:


#DRAFT functions display
intervals = ['5m','15m','30m','1h','2h','4h','8h','12h','1d','3d','1w']        

## return a list of timeframes per ta indicator taname  for specific sympbol
#series to list 
def srtolist(final:pd,taname)->[]:
    #ta = globals()[taname]
    array = final[taname].values
    array= str(array[0]).split(",")
    
    #for k in range(0,len(array)):
            #print (k)
            #print (f" {taname} = {array[k]} \n")
    return array

# list to string 
def ltos (mylist)->str:
    mystring = ''
    for x in mylist:
        if not mystring:
            mystring+=x
        else:
            mystring+=","+x
    return mystring

def get_key(my_dict,val):
        keylist = []
        for key, value in my_dict.items():
             if val in value:
                keylist.append(key)
        return keylist
    
def to_df(final:pd ,func,name):
    listb = func(final)
    res = list(sorted({ele for val in listb.values() for ele in val}))
    reversetable = []
    for indi in res:
        ttuple = (indi,ltos(get_key(listb,indi)))
        reversetable.append(ttuple)
    dframe = pd.DataFrame(reversetable,columns=['Indicator',name])
    return dframe
                                                
    
def tops(final:pd)-> pd :
    #intervals = ['5m','15m','30m','1h','2h','4h','8h','12h','1d','3d','1w']        
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['top'+i]
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : query.append(i)
        toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    

    return toplist


def uptrend(final:pd)-> pd :
    #intervals = ['1h','2h','4h','8h','12h','1d','3d','1w']        
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['uptrend'+i]
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    

    return toplist

def downtrend(final:pd)-> pd :
    #intervals = ['1h','2h','4h','8h','12h','1d','3d','1w']        
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['downtrend'+i]
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    

    return toplist

def shift(final:pd)-> pd :
    #intervals = ['1h','2h','4h','8h','12h','1d','3d','1w']        
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['shift'+i]
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    

    return toplist

def dshift(final:pd)-> pd :
    #intervals = ['1h','2h','4h','8h','12h','1d','3d','1w']        
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['dshift'+i]
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    

    return toplist

def topr(final:pd)-> pd :
    #intervals = ['5m','15m','30m','1h','2h','4h','8h','12h','1d','3d','1w']        
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['topr'+i]
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    

    return toplist

def bottomr(final:pd)-> pd :
    #intervals = ['1h','2h','4h','8h','12h','1d','3d','1w']        
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['bottomr'+i]
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    

    return toplist

def bottoms(final:pd)-> pd :
    #intervals = ['1h','2h','4h','8h','12h','1d','3d','1w']        
    query = []
    bottomlist ={}
    for i in intervals:
        var = globals()['bottom'+i]
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            bottomlist[i]=test['Indicator'].tolist()
 
    return bottomlist

def room(final:pd)-> pd :
    #intervals = ['1h','2h','4h','8h','12h','1d','3d','1w']        
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['room'+i]
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    

    return toplist

def roomr(final:pd)-> pd :
    #intervals = ['1h','2h','4h','8h','12h','1d','3d','1w']        
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['roomr'+i]
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    

    return toplist

def rsiup(final:pd)-> pd :
    #intervals = ['1h','2h','4h','8h','12h','1d','3d','1w']          
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['rule'+i+'RSIup']
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    

    return toplist

def rsidown(final:pd)-> pd :
    #intervals = ['1h','2h','4h','8h','12h','1d','3d','1w']         
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['rule'+i+'RSIdown']
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    

    return toplist

def stover(final:pd)-> pd :
    #intervals = ['1h','2h','4h','8h','12h','1d','3d','1w']          
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['rule'+i+'over']
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    

    return toplist

def stunder(final:pd)-> pd :
    #intervals = ['1h','2h','4h','8h','12h','1d','3d','1w']          
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['rule'+i+'under']
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    

    return toplist

def stkup(final:pd)-> pd :
    #intervals = ['1h','2h','4h','8h','12h','1d','3d','1w']          
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['rule'+i+'kup']
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    

    return toplist

def stkdown(final:pd)-> pd :
    #intervals = ['1h','2h','4h','8h','12h','1d','3d','1w']          
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['rule'+i+'kdown']
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    

    return toplist


def stkcross(final:pd)-> pd :
    #intervals = ['1h','2h','4h','8h','12h','1d','3d','1w']          
    query = []
    toplist ={}
    for i in intervals:
        var = globals()['stk'+i+'cross']
        test = final.loc[(var)]
        #print(test[['Indicator']])
        if len(test.index) > 0 : 
            query.append(i)
            toplist[i]=test['Indicator'].tolist()
        #print(query)
        #print(toplist)    
    return toplist


def buyzones(final:pd)->pd :
    ## buy zone based on rsi up and non top 
    toplist ={}
    indicators=final['Indicator'].tolist()
    for indicator in indicators:
        buy =[]
        test1 = final.loc[(newframe['Indicator'] == indicator)]
        lrsiup = srtolist(test1,'rsiup')
        lroom = srtolist(test1,'room')
        lroomr = srtolist(test1,'roomr')
        for item in lrsiup :
            if item in lroomr:
                if item in lroom:
                    buy.append(item)
        toplist[indicator] = ltos(buy)
        #better keep it as list for manipulation made it string for display
    #print(toplist)
    dframe=pd.DataFrame(toplist.items(), columns=['Indicator', 'buyzones'])
    #dframe = pd.DataFrame.from_dict(toplist,orient='index',columns=['Indicator','buyzones'])
    #tseries = pd.Series(toplist)
    #dframe.index.name='Indicator'
    #tseries.reset_index()
    #dframe = pd.DataFrame(tseries,columns=['Indicator','buyzones'])
    #dframeclomuns['Indicator','buyzones']
    final=final.merge(dframe ,how="left",on="Indicator")
    return final
  


#not used but defined for later 
def mergetables(final:pd,*arg)->pd:
    indicators=final[['Indicator']]
    for i in range(0,len(arg)):
        test = indicators.merge(arg[i] ,how="left",on="Indicator")
        indicators = test
    return test

# call the function and merge the tables 
def callfun(final:pd,*var)->pd:
    indicators=final[['Indicator']]
    for i in range(0,len(var)):
        test=to_df(final ,globals()[var[i]],var[i])
        indicators=indicators.merge(test ,how="left",on="Indicator")
    return indicators


def wrjson(wl,name):
    test = json.dumps(wl)
    with open(name, "w") as wlf:
        json.dump(test, wlf,indent=4)
        
def frtolist(final:pd)->[]:
    whitelist = []
    wl = []
    whitelist = final['Indicator'].tolist()
    for i in range(0 ,len(whitelist)):
        #print (str(whitelist[i].replace('USDT',"/USDT")))
        wl.append(str(whitelist[i].replace('USDT',"/USDT")))
    return wl
            


# In[15]:



newframe=callfun(final,'stkcross','rsiup','rsidown','tops','bottoms','topr','bottomr','stover','room','roomr','stunder','stkup','stkdown','uptrend','dshift','downtrend','shift')

#newframe=newframe.sort_values(by=['downtrend','dshift','bottoms'],ascending=[True,True,True])
newframe=newframe.sort_values(by=['bottoms','stkup','bottomr'],ascending=[False,True,False])
newframe=newframe.sort_values(by=['bottomr','bottoms','shift'],ascending=[False,False,False])
newframe=newframe.sort_values(by=['bottomr','bottoms','rsidown'],ascending=[False,False,False])
newframe=newframe.sort_values(by=['rsiup','bottoms','bottomr'],ascending=[False,False,False])

newframe.to_csv("allindi"+str(date.today())+".csv")

test = newframe.loc[(newframe['downtrend'] <= '1h')]
test = newframe.loc[(newframe['bottomr'] >= '12h')]
#test = newframe.loc[(newframe['rsidown'] >= '2h')]
#test = newframe.loc[(newframe['rsiup'] >= '1,2')]
#test = newframe.loc[(newframe['bottoms'] >= '2h,4h') & (newframe['rsiup'] >= '1,2')]
#test = newframe.loc[(newframe['bottoms'] >= '12h') & (newframe['bottomr'] >= '1') ]
#test = newframe.loc[(newframe['bottomr'] >= '1') & (newframe['rsiup'] >= '1')]
#test = newframe.loc[(newframe['bottomr'].notnull()) &  (newframe['rsidown'].notnull()) ]
#test = newframe.loc[(newframe['bottomr'].notnull()) &  (newframe['shift'].notnull()) ]
#test = newframe.loc[(newframe['bottoms'].notnull()) & (newframe['uptrend'].notnull()) &  (newframe['bottomr'].notnull()) ]
#test = newframe.loc[(newframe['bottoms'].notnull()) &  (newframe['stkup'].notnull()) & (newframe['rsidown'].isnull()) & (newframe['tops'] <='4h') & (newframe['topr'].isnull()) ]
#test = newframe.loc[(newframe['bottomr'].notnull()) &  (newframe['stkup'].notnull()) & (newframe['topr'].isnull()) ]

#test = newframe.loc[(newframe['bottoms'].notnull()) &  (newframe['topr'].isnull()) & (newframe['bottoms'].notnull()) ]

#test = newframe.loc[(newframe['stkdown'].isnull()) &  (newframe['bottoms'].notnull()) ]
#test = newframe.loc[(newframe['stunder'] <= '3d') &  (newframe['rsiup'] <= '1d') ]
#test = newframe.loc[(newframe['shift'].notnull()) &  (newframe['uptrend'].notnull()) ]

#test = newframe.loc[(newframe['shift'].isnull()) &  (newframe['uptrend'].isnull()) ]

#test = newframe.loc[(newframe['stunder'] <= '3d') &  (newframe['rsiup'] <= '1d') & (newframe['rsidown'].isnull())]
#test = newframe.loc[(newframe['rsiup'] <= '1d') & (newframe['rsidown'].isnull())]

#test = newframe.loc[(newframe['rsiup'].notnull()) & (newframe['bottoms'].notnull())  & (newframe['uptrend'].notnull())]


test = newframe.loc[(newframe['rsiup'] <= '8h') & (newframe['bottoms'].notnull())]

test = newframe.loc[(newframe['rsiup'] <= '1d') & (newframe['stover'].notnull())]

#test = newframe.loc[(newframe['rsidown'].isnull()) &  (newframe['stkup'].notnull()) ]
#test = newframe.loc[(newframe['rsidown'].notnull()) &  (newframe['stkup'].notnull()) ]
#test = newframe.loc[(newframe['rsidown'].notnull()) &  (newframe['bottoms'].notnull())]

#test = newframe.loc[(newframe['rsidown'].isnull())]

test = newframe.loc[(newframe['rsiup'] >= '1h,2h,4h') ]


#test = newframe.loc[(newframe['bottomr'] <= '12h') &  (newframe['bottoms'] <= '12h') ]
#test = newframe.loc[(newframe['bottomr'] >= '1d') & ((newframe['stover'].notnull()) | (newframe['stkcross'].notnull()))]
#test = newframe.loc[(newframe['bottoms'] >= '1d') & ((newframe['stover'].notnull()) | (newframe['stkup'].notnull()))]
#test = newframe.loc[(newframe['uptrend'].notnull()) &  (newframe['rsidown'].isnull()) ]

#test = newframe.loc[ (newframe['stkup'].notnull()) & (newframe['bottomr'] >= '2h') & ((newframe['stover'].notnull()) | (newframe['stkcross'].notnull()))]

#test = newframe.loc[  ((newframe['stover']>= '4h') | (newframe['stkcross']>='12h'))]

#test = newframe.loc[  ((newframe['stover']>= '4h') | (newframe['stkcross']>='12h')) & (newframe['bottomr'].notnull())]




#test = newframe.loc[(newframe['bottoms'] >= '1d') & ((newframe['stover'].notnull()) | (newframe['stkcross'].notnull()))]

#test = newframe.loc[(newframe['tops'].isnull()) &  (newframe['rsidown'].isnull()) ]


#test = newframe.loc[(newframe['stkup'].notnull()) &  (newframe['bottoms'].notnull()) & (newframe['tops'].isnull()) ]

#test = newframe.loc[(newframe['rsiup'] <= '1w') & (newframe['bottoms'] >= '4h') &  (newframe['shift'].notnull())]


#test = newframe.loc[(newframe['shift'].notnull()) &  (newframe['bottoms'].notnull()) ]

#test = newframe.loc[(newframe['rsidown'].isnull()) | (newframe['rsidown'] <= '1') ]

#test = newframe.loc[(newframe['shift'] >= '1h') &  (newframe['dshift'].isnull()) & (newframe['downtrend'].isnull()) ]
#test = newframe.loc[(newframe['shift'] >= '1h') &  (newframe['dshift'].isnull()) ]

#test = newframe.loc[(newframe['shift'] >= '12h,1d,3d')]

newframe

test


# In[16]:


test3 = buyzones(test)
test3


# In[17]:


## Indicator check
test1 = newframe.loc[(newframe['Indicator'] == 'WINUSDT')]
test1


# In[18]:


#list(set(list1 + list2))

test2 = newframe.loc[(newframe['Indicator'] == 'BTCUSDT')]['rsidown']
test2
buy =[]
lstkup =srtolist(test1,'stkup')
lrsiup = srtolist(test1,'rsiup')
lshift = srtolist(test1,'shift')
lroom = srtolist(test1,'room')
lroomr = srtolist(test1,'roomr')
luptrend = srtolist(test1,'uptrend')

#luptrend=list(set(luptrend + lshift))

for item in lrsiup :
    if item in lroomr:
        if item in lroom:
                buy.append(item)
  
print(buy)


# In[19]:


# Indicators for the week shift

newframe=callfun(wshift,'stkcross','rsiup','rsidown','tops','bottoms','topr','bottomr','stover','room','roomr','stunder','stkup','stkdown','uptrend','dshift','downtrend','shift')

#newframe=newframe.sort_values(by=['downtrend','dshift','bottoms'],ascending=[True,True,True])
newframe=newframe.sort_values(by=['bottoms','stkup','bottomr'],ascending=[False,True,False])

newframe=buyzones(newframe)
newframe


# In[20]:


# Indicators for the 3d Cross
newframe=callfun(macd1dd,'stkcross','rsiup','rsidown','tops','bottoms','topr','bottomr','stover','room','roomr','stunder','stkup','stkdown','uptrend','dshift','downtrend','shift')
newframe=newframe.sort_values(by=['bottomr','bottoms','rsidown'],ascending=[False,False,False])
test = newframe.loc[(newframe['shift'].notnull()) & (newframe['uptrend'].notnull()) &  (newframe['bottomr'].notnull()) ]
#test = newframe.loc[(newframe['uptrend'].notnull()) &  (newframe['bottoms'].notnull()) ]
#test = newframe.loc[(newframe['stkup'].notnull()) &  (newframe['bottoms'].notnull()) & (newframe['tops'].isnull()) ]
#test = newframe.loc[ (newframe['stkup'].notnull()) & (newframe['bottomr'] >= '2h') & ((newframe['stover']>= '2h') | (newframe['stkcross']>='1d'))]


test


# In[21]:


## MACD 3d gap and rsi up
newframe=callfun(macd3dd,'stkcross','rsiup','rsidown','tops','bottoms','topr','bottomr','stover','room','roomr','stunder','stkup','stkdown','uptrend','dshift','downtrend','shift')
newframe=newframe.sort_values(by=['bottomr','bottoms','rsidown'],ascending=[False,False,False])
test = newframe.loc[(newframe['shift'].notnull()) & (newframe['uptrend'].notnull()) &  (newframe['bottomr'].notnull()) ]
test = newframe.loc[(newframe['uptrend'].notnull()) &  (newframe['bottoms'].notnull()) ]
test = newframe.loc[(newframe['stkup'].notnull()) &  (newframe['bottoms'].notnull()) & (newframe['tops'].isnull()) ]
#test = newframe.loc[ (newframe['stkup'].notnull()) & (newframe['bottomr'] >= '2h') & ((newframe['stover']>= '2h') | (newframe['stkcross']>='3d'))]
test = newframe.loc[(newframe['stkup'].notnull()) &  (newframe['bottoms'].notnull()) ]
#test = newframe.loc[(newframe['rsiup'] > newframe['rsidown']) ]

#test = newframe.loc[(newframe['rsiup'].notnull()) &  (newframe['rsidown'] >= "1w" )]



test=buyzones(test)
test


# In[22]:


## MACD gap 1 Week and rsi up
newframe=callfun(macd1wd,'stkcross','rsiup','rsidown','tops','bottoms','topr','bottomr','stover','room','roomr','stunder','stkup','stkdown','uptrend','dshift','downtrend','shift')
newframe=newframe.sort_values(by=['bottomr','bottoms','rsidown'],ascending=[False,False,False])
test = newframe.loc[(newframe['shift'].notnull()) & (newframe['uptrend'].notnull()) &  (newframe['bottomr'].notnull()) ]
test = newframe.loc[(newframe['uptrend'].notnull()) &  (newframe['bottoms'].notnull()) ]
#test = newframe.loc[(newframe['stkup'].notnull()) &  (newframe['bottoms'].notnull()) & (newframe['tops'].isnull()) ]
#test = newframe.loc[ (newframe['stkup'].notnull()) & (newframe['bottomr'] >= '2h') & ((newframe['stover']>= '2h') | (newframe['stkcross']>='3d'))]
#test = newframe.loc[(newframe['uptrend'].notnull()) &  (newframe['rsidown'].isnull()) ]
#test = newframe.loc[(newframe['stkup'].notnull()) &  (newframe['bottoms'].notnull()) ]
#test = newframe.loc[(newframe['dshift'].isnull()) &  (newframe['bottoms'].notnull()) ]


test=buyzones(test)
#test=test.loc[test['buyzones']>='1h']
test


# In[39]:


## MACD gap 1 Week stk cross
newframe=callfun(stk1w,'stkcross','rsiup','rsidown','tops','bottoms','topr','bottomr','stover','room','roomr','stunder','stkup','stkdown','uptrend','dshift','downtrend','shift')
newframe=newframe.sort_values(by=['bottomr','bottoms','rsidown'],ascending=[False,False,False])
test = newframe.loc[(newframe['shift'].notnull()) & (newframe['uptrend'].notnull()) &  (newframe['bottomr'].notnull()) ]
#test = newframe.loc[(newframe['uptrend'] < '1d') &  (newframe['bottoms'].notnull()) ]
#test = newframe.loc[(newframe['stkup'].notnull()) &  (newframe['bottoms'].notnull()) & (newframe['tops'].isnull()) ]
#test = newframe.loc[ (newframe['stkup'].notnull()) & (newframe['bottomr'] >= '2h') & ((newframe['stover']>= '2h') | (newframe['stkcross']>='3d'))]
#test = newframe.loc[(newframe['uptrend'].notnull()) &  (newframe['rsidown'].isnull()) ]
#test = newframe.loc[(newframe['stkup'].notnull()) &  (newframe['bottoms'].notnull()) ]
#test = newframe.loc[(newframe['dshift'].isnull()) &  (newframe['bottoms'].notnull()) ]
#test = newframe.loc[(newframe['downtrend'].isnull()) ]



test=buyzones(test)
#test=test.loc[test['buyzones']>='1h']
test


# In[40]:


#hihi5macd1w
## MACD gap 1 Week and hihi5m
newframe=callfun(hihi5macd1w,'stkcross','rsiup','rsidown','tops','bottoms','topr','bottomr','stover','room','roomr','stunder','stkup','stkdown','uptrend','dshift','downtrend','shift')
newframe=newframe.sort_values(by=['bottomr','bottoms','rsidown'],ascending=[False,False,False])
test = newframe.loc[(newframe['shift'].notnull()) & (newframe['uptrend'].notnull()) &  (newframe['bottomr'].notnull()) ]
test = newframe.loc[(newframe['uptrend'].notnull()) &  (newframe['bottoms'].notnull()) ]
#test = newframe.loc[(newframe['stkup'].notnull()) &  (newframe['bottoms'].notnull()) & (newframe['tops'].isnull()) ]
#test = newframe.loc[ (newframe['stkup'].notnull()) & (newframe['bottomr'] >= '2h') & ((newframe['stover']>= '2h') | (newframe['stkcross']>='3d'))]
#test = newframe.loc[(newframe['uptrend'].notnull()) &  (newframe['rsidown'].isnull()) ]
#test = newframe.loc[(newframe['stkup'].notnull()) &  (newframe['bottoms'].notnull()) ]
#test = newframe.loc[(newframe['dshift'].isnull()) &  (newframe['bottoms'].notnull()) ]


test=buyzones(test)
#test=test.loc[test['buyzones']>='1h']
test


# In[41]:


white =  frtolist(test)
print(white)      
wrjson(white,"whitelistfile.json")


# In[45]:


white =  frtolist(higherhi5)
print(white)      
wrjson(white,"hihi5.json")


# In[25]:


test = newframe.loc[(newframe['Indicator'] == 'BTCUSDT')]
print(test['Indicator'])


# # Create a function to filter the above 
#     - things at the bottom and going up then go  down in timeframes 
#     - 1d Bottom 
#         - check 12 Bottom either going towards it
#         - At the bottom of 12 
#         - Already moved from the bottom of 12 
#     
