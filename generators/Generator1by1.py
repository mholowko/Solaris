# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:42:31 2019

@author: hol428
"""
import xlwt

#Sequence of the template to be modified
T='tttaagaaggagatatacat'

#length of the generated sequence, it will generate 4*N sequences
N = len(T)

#sequence collector list
sequences = [T]

for index,base in enumerate(T):
    if base == 'a':
        Ttemp = T[:index] + 'c' + T[index + 1:]
        sequences.append(Ttemp)
        Ttemp = T[:index] + 'g' + T[index + 1:]
        sequences.append(Ttemp)
        Ttemp = T[:index] + 't' + T[index + 1:]
        sequences.append(Ttemp)
    elif base == 'c':
        Ttemp = T[:index] + 'a' + T[index + 1:]
        sequences.append(Ttemp)
        Ttemp = T[:index] + 'g' + T[index + 1:]
        sequences.append(Ttemp)
        Ttemp = T[:index] + 't' + T[index + 1:]
        sequences.append(Ttemp)   
    elif base == 't':
        Ttemp = T[:index] + 'c' + T[index + 1:]
        sequences.append(Ttemp)
        Ttemp = T[:index] + 'g' + T[index + 1:]
        sequences.append(Ttemp)
        Ttemp = T[:index] + 'a' + T[index + 1:]
        sequences.append(Ttemp)      
    elif base == 'g':
        Ttemp = T[:index] + 'c' + T[index + 1:]
        sequences.append(Ttemp)
        Ttemp = T[:index] + 'a' + T[index + 1:]
        sequences.append(Ttemp)
        Ttemp = T[:index] + 't' + T[index + 1:]
        sequences.append(Ttemp) 

#create Excel workbook to save the sequences to
book = xlwt.Workbook()
sh = book.add_sheet('Sequences')
      
for el,row in zip(sequences,range(0,80)):
    sh.write(row,0,'Random_RBS' + '_' + str(row))
    sh.write(row,1,str(el))
    
#save the workbook
book.save('RBS1by1seq' + '.xls')
    