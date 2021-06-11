# -*- coding: utf-8 -*-
"""
Created on Wed 20 Sep 14:16:13 2019
@author: hol428
Script to generate random sequences of DNA
"""

import random
import xlwt

#define function that generates a stretch of DNA of n length 
def nrandom (length=6,char=['A','T','G','C']):
    return ''.join(random.choice(['A','T','G','C']) for _ in range(length))

sequences = []
tester = []
seqN = 4096

#sequence generator
for seq in range (0,seqN):
    #regeneration of random sequences on each pass
    
    seq2 = nrandom(6)
    
    while seq2 in tester:
        seq2 = nrandom(6)
    
    tester.append(seq2)
    
    sequences.append(nrandom(6))
    print('Generating sequence number: ' + str(seq))
    
#create Excel workbook to save the sequences to
book = xlwt.Workbook()
sh = book.add_sheet('Sequences')

#generate the sequences, however many you like
for el,row in zip(tester,range(0,seqN)):
    sh.write(row,0,'Random_RBS' + '_' + str(row))
    sh.write(row,1,str(el))

#save the workbook
book.save('RBSseq' + '.xls')