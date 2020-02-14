# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:14:04 2020

@author: hol428
"""

import pandas as pd
import xlwt

df = pd.read_excel("First Plate 050220 (Rep 3 of 3).xlsx",sheet_name='ODGFP')
df.drop(["Time"],axis=1,inplace=True)

zupa = df.min(axis=0)

time = [60,120,180,240]

rates =[]

for t in time:
    for column,value in zupa.iteritems():
        start = df[df[column]==value].index.values.astype(int)[0]
        end = start + (t/10)
        rate = (df.loc[start:end, [column]].sum() / t)
        rates.append(float(rate[0]))
        print(rates)

book = xlwt.Workbook()
sh = book.add_sheet('rates')

for el,row in zip(rates,range(0,len(rates))):
    sh.write(row,0,'sss' + '_' + str(row))
    sh.write(row,1,str(el))

book.save('Rates' + '.xls')

    