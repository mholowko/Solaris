# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:16:13 2019

@author: hol428
Script to generate random sequences of DNA
"""

import random
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
import xlwt

#define function that generates a stretch of DNA of n length 
def nrandom (length=6,char=['A','T','G','C']):
    return ''.join(random.choice(['A','T','G','C']) for _ in range(length))

#set the parts of DNA sequence that do not get randomized
defregion1 = 'AAAAAAAAAAAAAAAAAA'
defregion2 = 'AAAAAAAAAAAAAAAAAA'
defregion3 = 'AAAAAAAAAAAAAAAAAA'
defregion4 = 'AAAAAAAAAAAAAAAAAA'

#empty list for storing the sequences
sequences = []

for seq in range (0,1001):
    sequences.append(Seq(defregion1 + nrandom(6) + defregion2 + nrandom(6) + defregion3 + nrandom(6)
    + defregion4,generic_dna))

#create Excel workbook to save the sequences to
book = xlwt.Workbook()
sh = book.add_sheet('Sequences')

#generate the sequences, however many you like
for el,row in zip(sequences,range(0,1001)):
    sh.write(row,0,str(el))

#save the workbook
book.save('SequencesforTwist.xls')