#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import subprocess
import schedule
import time
import datetime
from datetime import date


#cwd = os.getcwd()
#print(cwd)


# In[2]:


subprocess.call("pwd")
#os.chdir(cwd)


# In[3]:


#subprocess.call("./Easyband-Entries.py")


# In[4]:


#subprocess.call("./EASYCAP.py")


# In[5]:


thistime = datetime.datetime.now() 
thisdate = date.today()

def easycap():
    print("EasyCap")
    print(str(thistime))
    print(str(thisdate))
    subprocess.call("./EASYCAP.py")
    
def easyband():
    print("EasyBand")
    print(str(thistime))
    print(str(thisdate))
    subprocess.call("./Easyband-Entries.py")
    
def strsi3d1h():
    print("STRSI3D1H")
    print(str(thistime))
    print(str(thisdate))
    subprocess.call("./STRSI3D1H.py")
    
def mldata():
    print("MLDATA")
    print(str(thistime))
    print(str(thisdate))
    subprocess.call("./datacollector.py")
    
def fixentry():
    print("fixentry")
    print(str(thistime))
    print(str(thisdate))
    subprocess.call("./FixEntries.py")
    
schedule.every(7).days.do(easycap)
schedule.every(24).hours.do(easyband)
#schedule.every(1).hour.do(strsi3d1h)
schedule.every(1).hour.do(fixentry)
schedule.every(15).minutes.do(mldata)



  
while True:
    schedule.run_pending()
    time.sleep(1)


# In[ ]:





# In[ ]:




