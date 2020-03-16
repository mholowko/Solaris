# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:31:14 2020

@author: hol428
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


#read the res(ults) file and the loc(ations) file
res1 =  pd.read_excel('First Plate 300120 (Rep 1 of 3).xlsx',sheet_name='ODGFP',index_col=0)
res2 =  pd.read_excel('First Plate 310120 (Rep 2 of 3).xlsx',sheet_name='ODGFP',index_col=0)
res3 =  pd.read_excel('First Plate 050220 (Rep 3 of 3).xlsx',sheet_name='ODGFP',index_col=0)
res=[res1,res2,res3]
loc = pd.read_excel('1st Primer Plate 11122019.xlsx')

#change rows from A01 to A1 format in the locations to match results file
for index, row in loc.iterrows():
    if int(row['Well Position'][1]) == 0:
        loc.at[index,'Well Position'] = row['Well Position'][0] + row['Well Position'][2] 

#change wells into 'Sequence'  or 'Sequence Name'

for i in res:
    for col in i.columns:
        for index, row in loc.iterrows():
            if row['Well Position'] == col:
                    m = re.search('ATGTATA(.+?)TCTTAAA', row['Sequence'].replace(" ",""))
                    if m:
                        i.rename(columns={ col : m.group()}, inplace=True)
                    else:
                        i.rename(columns={ col : row['Sequence'].replace(" ","")}, inplace=True)

writer = pd.ExcelWriter('features&labels.xlsx', engine='xlsxwriter')
res1.loc[400].to_excel(writer, sheet_name='Sheet1')
res2.loc[380].to_excel(writer, sheet_name='Sheet2')
res3.loc[360].to_excel(writer, sheet_name='Sheet3')
writer.save()