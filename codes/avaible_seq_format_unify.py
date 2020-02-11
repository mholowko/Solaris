import numpy as np
import pandas as pd

first_round_seq = pd.read_csv('../data/first_round_seq.csv')

if not (first_round_seq['RBS'].str.len() == 20).all():
    mask = first_round_seq['RBS'].str.len() == 6
    first_round_seq['RBS'].values[mask] = 'TTTAAGA' + first_round_seq['RBS'].values[mask] + 'TATACAT'
assert (first_round_seq['RBS'].str.len() == 20).all()

first_round_seq['RBS'] = first_round_seq['RBS'].str.upper()
first_round_seq['RBS6'] = first_round_seq['RBS'].str[7:13]
#first_round_seq.to_csv('../../data/first_round_seq.csv')