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

#We will cycle three times to generate three lists, each with different randomized regions
#ver1 - all three regions; ver2 - -35 and -10 randomized; ver3 - -35 and RBS randomized

#definition of regions for each version:

defreg = {'ver1':['CGTATTGGGCGCCAGGGTGGTTTTTCTTTTCACCAGTGAGACGGGCAACAGCTGATTGC', 'R', 'GCTAGCTCAGTCCTAGG', 'R', 'GCTAGCTCTAGAGAAA', 'R', 'AAATACTAGATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGA'],
          'ver2':['CGTATTGGGCGCCAGGGTGGTTTTTCTTTTCACCAGTGAGACGGGCAACAGCTGATTGC', 'R', 'GCTAGCTCAGTCCTAGG', 'R', 'GCTAGCTCTAGAGAAA', 'GAGGAG', 'AAATACTAGATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGA'],
          'ver3':['CGTATTGGGCGCCAGGGTGGTTTTTCTTTTCACCAGTGAGACGGGCAACAGCTGATTGC', 'R', 'GCTAGCTCAGTCCTAGG', 'TATAAT', 'GCTAGCTCTAGAGAAA', 'R', 'AAATACTAGATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGA']}


for key,i in zip(defreg.keys(),range(1,4)):
    
    #empty list for storing the sequences
    sequences = []
    
    #sequence generator
    for seq in range (0,101):
        #regeneration of random sequences on each pass
        tempseq=''
        for j in range (0,7):
            if defreg[key][j] == 'R':
                tempseq += nrandom(6)
            else:
                tempseq += defreg[key][j]
        sequences.append(Seq(tempseq,generic_dna))
        
    #create Excel workbook to save the sequences to
    book = xlwt.Workbook()
    sh = book.add_sheet('Sequences')
    
    #generate the sequences, however many you like
    for el,row in zip(sequences,range(0,101)):
        sh.write(row,0,'MLEXPRver' + str (i) + '_' + str(row))
        sh.write(row,1,str(el))
    
    #save the workbook
book.save('SequencesforTwistver' + str(i) + '.xls')