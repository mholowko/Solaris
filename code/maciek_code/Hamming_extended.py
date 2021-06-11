# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:00:48 2021

@author: hol428
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:16:38 2021

@author: hol428
"""

from scipy.spatial.distance import hamming
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def separate(sequence):
    sequence2 = []
    for elem in sequence:
        sequence2.append(elem)
    return(sequence2)


df1 = pd.read_excel("Designs.xlsx", sheet_name = "TOP15")
df2 = pd.read_excel("Designs.xlsx", sheet_name = "All")

instances1 = []
instances2 = []

for seq in df1['Core'].to_list():
    seq = str(seq)
    instances1.append(seq.upper())
    
for seq in df2['Core'].to_list():
    seq = str(seq)
    instances2.append(seq.upper())

sequences = []
distance = []
values = []
number=[]
difference = []
violindf = pd.DataFrame(columns=['Parent','Neighbour'])

for seq1 in instances1[0:1]:
    print(instances1[0:1])
    seq1sep = separate(seq1)
    container = []
    for seq2 in instances2:
        seq2sep = separate(seq2)
        for h in range (6):
            h += 1
            if hamming(seq1sep,seq2sep)*len(seq1sep) == h:
                container.append(seq2)
                distance.append(h)
                values.append(df2.loc[df2['Core'] == container[-1]]['TIR'].values.astype(int)[0])
    sequences.append(seq1)
    number.append(len(container))
    print(distance)
    print(values)

data = pd.DataFrame ({'TIR': values, 'Distance': distance}, columns = ['TIR','Distance'])
print(data.head(60))
# data = data.sort_values(by=['TIR'])

ax = sns.swarmplot(data=data, x='Distance', y='TIR', palette='viridis')
ax.axhline(78,color='black')
plt.show()