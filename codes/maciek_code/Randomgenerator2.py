import random
import xlwt

#define function that generates a stretch of DNA of n length 
def nrandom (length=6,char=['A','T','G','C']):
    return ''.join(random.choice(['A','T','G','C']) for _ in range(length))

sequences = []
seqN = 88

#sequence generator
for seq in range (0,seqN):
    #regeneration of random sequences on each pass
    
    seq2 = nrandom(6)
    
    while seq2 in sequences:
        seq2 = nrandom(6)
    
    sequences.append(seq2)
    
#create Excel workbook to save the sequences to
book = xlwt.Workbook()
sh = book.add_sheet('Sequences')

#generate the sequences, however many you like
for el,row in zip(sequences,range(0,seqN)):
    sh.write(row,0,'Random_RBS' + '_' + str(row))
    sh.write(row,1,str(el))

#save the workbook
book.save('RBSseq' + '.xls')
