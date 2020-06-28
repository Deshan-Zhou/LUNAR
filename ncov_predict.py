import pandas as pd
from settings import *
parm = get_settings()
pre_data = pd.read_csv(parm.path + 'drug_protein.csv', encoding='utf-8',header=0,index_col=0)

#print(pre_data)
Chloroquine_j = []                     #DB00608
Darunavir_j = []                       #DB01264
Hydroxychloroquine_j = []              #DB01611

chl_dg = []
dar_dg = []
hyd_dg = []

########################Chloroquine#########################################
data = pd.read_csv(parm.path + 'results.csv', encoding='utf-8',header=0,index_col=0)
#Find the targets with the highest confidence in these drugs, choose 2 for each drug
count = 0
while count < 2:
    #Chloroquine
    temp = data.loc['DB00608',:]
    max_temp = temp[temp == temp.max()].index
    count = count +1
    data.loc['DB00608', max_temp[0]] = 0
    Chloroquine_j.append(max_temp[0])

#Based on these targets, find the drugs with the highest confidence.
#These are the drugs that may have inhibitory effects on coronavirus.
count = 0
while count < 4:
    count = count + 1
    for pt in Chloroquine_j:
        temp = data.loc[:, pt]
        max_temp = temp[temp == temp.max()].index
        data.loc[max_temp[0]][pt] = 0
        print(max_temp[0])
        chl_dg.append(max_temp[0])


########################Darunavir#########################################
data = pd.read_csv(parm.path + 'results.csv', encoding='utf-8',header=0,index_col=0)
count = 0
while count < 2:
    #Darunavir
    temp = data.loc['DB01264',:]
    max_temp = temp[temp == temp.max()].index
    count = count +1
    data.loc['DB01264', max_temp[0]] = 0
    Darunavir_j.append(max_temp[0])

count = 0
while count < 4:
    count = count + 1
    for pt in Darunavir_j:
        temp = data.loc[:, pt]
        max_temp = temp[temp == temp.max()].index
        data.loc[max_temp[0]][pt] = 0
        print(max_temp[0])
        dar_dg.append(max_temp[0])


########################Hydroxychloroquine#########################################
data = pd.read_csv(parm.path + 'results.csv', encoding='utf-8',header=0,index_col=0)
count = 0
while count < 2:
    #Hydroxychloroquine
    temp = data.loc['DB01611',:]
    max_temp = temp[temp == temp.max()].index
    count = count +1
    data.loc['DB01611', max_temp[0]] = 0
    Hydroxychloroquine_j.append(max_temp[0])

count = 0
while count < 4:
    count = count + 1
    for pt in Hydroxychloroquine_j:
        temp = data.loc[:, pt]
        max_temp = temp[temp == temp.max()].index
        data.loc[max_temp[0]][pt] = 0
        print(max_temp[0])
        hyd_dg.append(max_temp[0])


print("Retargeting of chloroquine:",Chloroquine_j)
print("Retargeting of Darunavir:",Darunavir_j)
print("Retargeting of Hydroxychloroquine:",Hydroxychloroquine_j)
print("Drugs screened by Chloroquine:",set(chl_dg))
print("Drugs screened by Darunavir:",set(dar_dg))
print("Drugs screened by Hydroxychloroquine:",set(hyd_dg))