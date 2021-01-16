#!/usr/bin/env python
# coding: utf-8

# In[15]:


from padercontrib.database.fearless import Fearless
from paderbox.array import intervall
from collections import Counter
import numpy as np
import pandas as pd
import re
import os


# In[2]:


Fearless = Fearless()
FearlessData = Fearless.data
devList=list(FearlessData['datasets']['Dev_segment'].items())
trnList=list(FearlessData['datasets']['Train_segment'].items())
evalList=list(FearlessData['datasets']['Eval_segment'].items())
devSegLst=[]
trnSegLst=[]
evalSegLst=[]
for a,b in devList:
    devSegLst.append(b)

for a,b in trnList:
    trnSegLst.append(b)
    
for a,b in evalList:
    evalSegLst.append(b)


# In[7]:


dfS=pd.DataFrame(devSegLst)
dfT=pd.DataFrame(trnSegLst)
dfE=pd.DataFrame(evalSegLst)

dfS['audio_id']=np.nan
dfS['audio_idprefix']=np.nan

dfT['audio_id']=np.nan
dfT['audio_idprefix']=np.nan

dfE['audio_id']=np.nan
dfE['audio_idprefix']=np.nan

for i in range(len(evalSegLst)):
    dfE.loc[i,'audio_path']=dfE.loc[i,'audio_path']['observation']
    dfE.loc[i,'audio_id']=re.split(r"[ /.]+",dfE.iloc[i]['audio_path'])[8]
    dfE.loc[i,'audio_idprefix']=dfE.loc[i,'audio_id'].split('_')[0]
    
for i in range(len(trnSegLst)):
    dfT.loc[i,'audio_path']=dfT.loc[i,'audio_path']['observation']
    dfT.loc[i,'audio_id']=re.split(r"[ /.]+",dfT.iloc[i]['audio_path'])[8]
    dfT.loc[i,'audio_idprefix']=str(dfT.loc[i,'speaker_id'])+'-'+str(dfT.loc[i,'audio_id'])
for i in range(len(devSegLst)):
    dfS.loc[i,'audio_path']=dfS.loc[i,'audio_path']['observation']
    dfS.loc[i,'audio_id']=re.split(r"[ /.]+",dfS.iloc[i]['audio_path'])[8]
    dfS.loc[i,'audio_idprefix']=str(dfS.loc[i,'speaker_id'])+'-'+str(dfS.loc[i,'audio_id'])


# In[18]:


if not os.path.exists("/net/vol/vivekkan/experiments/fearless/dump/raw/devset"):
    os.mkdir("/net/vol/vivekkan/experiments/fearless/dump/raw/devset")
if not os.path.exists("/net/vol/vivekkan/experiments/fearless/dump/raw/evalset"):
    os.mkdir("/net/vol/vivekkan/experiments/fearless/dump/raw/evalset")
if not os.path.exists("/net/vol/vivekkan/experiments/fearless/dump/raw/trainset"):
    os.mkdir("/net/vol/vivekkan/experiments/fearless/dump/raw/trainset")


# In[19]:


def createFiles(dataF,pthMz):
    curDir=os.getcwd()
    chkDir='/net/vol/vivekkan/experiments/fearless/dump/raw'
    updDir=chkDir + '/' +pthMz
    os.chdir(updDir)
    tempDf=dataF
    if pthMz == 'devset' or pthMz == 'trainset':
        with open("wav.scp",'w',encoding = 'utf-8') as f:
            for i in range(len(tempDf)):
                x=tempDf.iloc[i]['audio_idprefix']+"  "+tempDf.iloc[i]['audio_path']
                f.write(x+'\n')
            f.close()
        os.system('sort wav.scp -o wav.scp')
    else:
        with open("wav.scp",'w',encoding = 'utf-8') as f:
            for i in range(len(tempDf)):
                x=tempDf.iloc[i]['audio_id']+"  "+tempDf.iloc[i]['audio_path']
                f.write(x+'\n')
            f.close()        
    if pthMz == 'devset' or pthMz == 'trainset':
        with open("text",'w',encoding = 'utf-8') as f:
            for i in range(len(tempDf)):
                x=tempDf.iloc[i]['audio_idprefix']+"  "+tempDf.iloc[i]['transcription']
                f.write(x+'\n')
            f.close()
            os.system('sort text -o text')
        with open("utt2spk",'w',encoding = 'utf-8') as f:
            for i in range(len(tempDf)):
                x=tempDf.iloc[i]['audio_idprefix']+"  "+tempDf.iloc[i]['speaker_id']
                f.write(x+'\n')
            f.close()
            os.system('sort utt2spk -o utt2spk')
        with open("spk2utt",'w',encoding = 'utf-8') as f:
            testNames=tempDf['speaker_id'].unique()
            for i in range(len(testNames)):
                testList=str(list(tempDf[tempDf['speaker_id']==testNames[i]]['audio_idprefix'])).replace('[','').replace(']','').replace(',','').replace("'","")
                testSpeaker=testNames[i]
                testConcat=''
                for j in range(len(testList)):
                    testConcat += testList[j]
                f.write(testSpeaker+' '+testConcat+'\n')    
            f.close()
            os.system('sort spk2utt -o spk2utt')
    else:
        with open("text",'w',encoding = 'utf-8') as f:
            for i in range(len(tempDf)):
                x=tempDf.iloc[i]['audio_id']
                f.write(x+'\n')
            f.close()
        with open("utt2spk",'w',encoding = 'utf-8') as f:
            for i in range(len(tempDf)):
                x=tempDf.iloc[i]['audio_id']
                f.write(x+'\n')
            f.close()
        with open("spk2utt",'w',encoding = 'utf-8') as f:
            for i in range(len(tempDf)):
                x=tempDf.iloc[i]['audio_id']
                f.write(x+'\n')
            f.close()
    os.chdir(curDir)


# In[20]:


createFiles(dfS,"devset")


# In[21]:


createFiles(dfT,"trainset")


# In[22]:


createFiles(dfE,"evalset")


# In[ ]:




