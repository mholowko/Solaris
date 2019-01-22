"""
Created on Tue Jan 22 17:16:13 2019

@author: hol428
"""

import random
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
import xlwt

def nrandom (size=18,char=['A','T','G','C']):
    return ''.join(random.choice(['A','T','G','C']) for _ in range(size))

sequences = []

for seq in range (0,1001):
    sequences.append(Seq(nrandom(),generic_dna))

book = xlwt.Workbook()
sh = book.add_sheet('Sequences')


for el,row in zip(sequences,range(0,1001)):
    sh.write(row,0,str(el))
    print(el)

book.save('SequencesforTwist.xls')
