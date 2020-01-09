# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:24:10 2020

@author: hol428
"""

import sys
import Bio
import xlwt

seq_record = Bio.SeqIO.read('Escherichia_coli_str_K-12_substr_MG1655_ASM584v2_genomic.gbff','gb')
cds_list_plus = []
cds_list_minus = []
for feature in seq_record.features:
    if feature.type == 'CDS':
        if feature.strand == -1:
            mystart = feature.location.end + 0
            myend = feature.location.end + 10 
            cds_list_minus.append((mystart,myend,-1))
        elif feature.strand == 1:
            mystart = feature.location.start - 10
            myend = feature.location.start + 0
            cds_list_plus.append((mystart,myend,1))
        else:
            sys.stderr.write("No strand indicated %d-%d. Assuming +\n" %
                              (mystart, myend))
            cds_list_plus.append((mystart,myend,1))


rbsplus = []
rbsminus = []

for item in cds_list_plus:
    rbsplus.append(str((seq_record.seq[item[0]:item[1]])))
    
for item in cds_list_plus:
    rbsminus.append(str((seq_record.seq[item[0]:item[1]].reverse_complement())))

rbs_full = rbsplus + rbsminus
print(rbs_full)

book = xlwt.Workbook()
sh = book.add_sheet('RBS')

for el,row in zip(rbs_full,range(0,len(rbs_full))):
    sh.write(row,0,'GenomicRBS' + '_' + str(row))
    sh.write(row,1,str(el))

book.save('RBSgenome' + '.xls')
    
