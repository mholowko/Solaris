# -*- coding: utf-8 -*-
"""
TIR calculator

by Maciej Holowko @CSIRO
"""

import pandas as pd

directory='../../data/First_round_results/'
Filename='Third Plate 161020 (Rep 1 of 6).xlsx'
Path = directory + Filename

OD = pd.read_excel(Path,sheet_name='OD')
GFP = pd.read_excel(Path,sheet_name='GFP')
GFPOD = OD.copy(deep=True)
i=0

OD.drop('Time',axis=1,inplace=True)

for col in OD.columns:
    for ind in GFPOD.index:
        print(col)
        print(ind)
        GFPOD[col][ind]=GFP[col][ind]/OD[col][ind]
        
for ind in GFPOD.index:
    GFPOD['Time'][ind]=i
    i+=10

GFPOD.set_index('Time',inplace=True)

with pd.ExcelWriter(Path, engine="openpyxl", mode='a') as writer:  
    GFPOD.to_excel(writer, sheet_name='GFPOD')

GFPOD.reset_index(drop=True,inplace=True)

minp = GFPOD.min(axis=0)

time = [240]

rates = pd.DataFrame(columns=['240','Comment'],index = list(GFPOD.columns))

for t in time:
    for column,value in minp.iteritems():
        start = GFPOD[GFPOD[column]==value].index.values.astype(int)[0]
        end = start + (t/10)
        rate = (GFPOD.loc[start:end, [column]].sum() / t)
        # print(start)
        # print(end)
        # print(rate)
        rates.at[column,str(t)] = float(rate[0])
        if rate[0]<0:
            rates.at[column,'Comment'] = 'Impossible value'
        elif rate[0]>100 and OD.max(axis=0)[column]<0.1:          
            rates.at[column,'Comment'] = 'Possibly an empty well'
        else:
            rates.at[column,'Comment'] = 'Correct value'

#print(rates)

with pd.ExcelWriter(Path, engine="openpyxl", mode='a') as writer:  
    rates.to_excel(writer, sheet_name='TIR')