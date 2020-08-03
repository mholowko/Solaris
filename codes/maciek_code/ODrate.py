# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:01:36 2020

@author: hol428
"""

import pandas as pd

directory='../../data/First_round_results/'
Filename='Second Plate 310720 (Rep 3 of 3, full).xlsx'
Path = directory + Filename


df = pd.read_excel(Path,sheet_name='ODGFP')
df2 = pd.read_excel(Path,sheet_name='OD')
df.drop(["Time"],axis=1,inplace=True)
df2.drop(["Time"],axis=1,inplace=True)

zupa = df.min(axis=0)

time = [60,120,180,240]

rates = pd.DataFrame(columns=['60','120','180','240','60OD','120OD','180OD','240OD'],index = list(df.columns))


for t in time:
    for column,value in zupa.iteritems():
        start = df[df[column]==value].index.values.astype(int)[0]
        end = start + (t/10)
        rate = (df.loc[start:end, [column]].sum() / t)

        rates.at[column,str(t)] = float(rate[0])
        
        rate2 = (df2.loc[start:end, [column]].sum() / t)

        rates.at[column,str(t)+'OD'] = float(rate2[0])        

print(rates)

rates.to_csv(path_or_buf=directory+'rates.csv')