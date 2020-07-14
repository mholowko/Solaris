# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:14:04 2020

@author: hol428
"""

import pandas as pd

directory='../../data/First_round_results/'
Filename='First Plate 050220 (Rep 5 of 6).xlsx'
Path = directory + Filename


df = pd.read_excel(Path,sheet_name='ODGFP')
df.drop(["Time"],axis=1,inplace=True)

zupa = df.min(axis=0)

time = [60,120,180,240]

rates = pd.DataFrame(columns=['60','120','180','240'],index = list(df.columns))


for t in time:
    for column,value in zupa.iteritems():
        start = df[df[column]==value].index.values.astype(int)[0]
        end = start + (t/10)
        rate = (df.loc[start:end, [column]].sum() / t)

        rates.at[column,str(t)] = float(rate[0])

print(rates)

rates.to_csv(path_or_buf=directory+'rates.csv')

    