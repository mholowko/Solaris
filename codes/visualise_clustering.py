
from nltk.metrics import distance 
import numpy as np
import pandas as pd 
import itertools
import math
from kernels_for_GPK import *
from embedding import Embedding
from sklearn_extra.cluster import KMedoids
from sklearn.manifold import TSNE
# import umap
# import os
import plotly.express as px
from sklearn.utils import shuffle
import argparse

# Nov/2020 Mengyan Zhang
# This script visualise the DNA sequences in clusterings.
# The goal is to visualise the recommendations to see 
# whether they are similar to each other, whether they cover the whole space etc.

# Includes:
# 1) Kmediods on the pre-computed distance based on the weighted degree kernel with shift
# 2) TSNE/UMAP
# 3) Scatter plot via plotly

# input parameters
parser = argparse.ArgumentParser(description='Visualisation of RBS Clusterings.')
parser.add_argument('new_round_sheet_name', default= 'gpbucb_alpha2_beta2', help = 'sheet name of batch_ucb.xlsx')

args = parser.parse_args()
new_round_sheet_name = str(args.new_round_sheet_name) # str 

# new_round_sheet_name = gpbucb_core__alpha2_beta2kernelNormTrue
# setting
random_state = 24
n_dim = 2 # dimension reduction 
scores = {}
ALL_DESIGN_SPACE = True
New_Round = True
distance_name = 'wd_shift_distance'

n_clusters = 6 # to be changed

def generate_design_space(known_rbs_set):
    # create all combos

    combos = [] # 20-base
    combos_6 = [] # 6-base

    # Setting
    char_sets = ['A', 'G', 'C', 'T']
    design_len = 6
    pre_design = 'TTTAAGA'
    pos_design = 'TATACAT'

    for combo in itertools.product(char_sets, repeat= design_len):
        combo = pre_design + ''.join(combo) + pos_design
        combos_6.append(''.join(combo))
        combos.append(combo)
        
    assert len(combos) == len(char_sets) ** design_len

    # df_design = pd.DataFrame()
    design_seq = [x for x in combos if x not in known_rbs_set]
    # df_design['RBS6'] = df_design['RBS'].str[7:13]

    return design_seq

# read data
Folder_Path = os.getcwd() # folder path might need to change for different devices
# Path = '../../../data/Results_Microplate_partialFalse_normFalse_formatSeq_logTrue.csv'
Path = '/data/Results_Microplate_partialTrue_normTrue_roundRep_formatSeq_logTrue.csv'
New_round_path = '/notebooks/rec_design/batch_ucb.xlsx'
plot_title = 'Round01_'+str(n_clusters)+ '_Medoids_TNSE_' + str(n_dim) + '_dim_' + str(ALL_DESIGN_SPACE)+ '_allSeq_' + str(New_Round) + '_newRound'
save_path = Folder_Path + '/data/Clustering/' + plot_title + '.html'

df = pd.read_csv(Folder_Path + Path)
# df['Group Code'] = df.Group.astype('category').cat.codes
known_data = np.asarray(df[['RBS', 'RBS6', 'Group', 'Pred Mean', 'AVERAGE']])
known_seq = np.asarray(df['RBS'])
print('Known_seq shape ', known_seq.shape)

if New_Round:
    df_new_round = pd.read_excel(Folder_Path + New_round_path, sheet_name= new_round_sheet_name)[['RBS', 'RBS6', 'Pred Mean']]
    df_new_round['Group'] = 'new rec'
    df = pd.concat([df, df_new_round])
    df.reset_index(inplace=True, drop=True)

# TODO: the parameters might need to change according to input new round data
# for example, if input data is generated without kernel normalisation, we may want to tune centering_flag and unit_norm_flag to False
kernel_instance = WD_Shift_Kernel(l=6, s=1,sigma_0=1,centering_flag=True,unit_norm_flag=True)


if ALL_DESIGN_SPACE:
    design_seq = generate_design_space(set(np.asarray(df['RBS'])))
    df_design = pd.DataFrame()
    df_design['RBS'] = design_seq
    df_design['RBS6'] = df_design['RBS'].str[7:13]
    df_design['Group'] = 'unknown'
    df = pd.concat([df_design, df])
    # df = shuffle(pd.concat([df_design, df]))
    df.reset_index(inplace=True, drop=True)
    print(df)

    all_seq = np.asarray(df['RBS'])
    distance = kernel_instance.distance(kernel_instance.__call__(all_seq))
    print('all seq distance: ', distance.shape)
else:
    # distance = WD_Shift_Kernel(features = known_seq, l = 6, s=1).distance_all
    distance = kernel_instance.distance(kernel_instance.__call__(known_seq))
    print('known seq distance: ', distance.shape)



def kmedoids(n_clusters = 6, random_state = 0, distance = None):
    # clustering
    kmedoids = KMedoids(n_clusters=n_clusters, metric = 'precomputed', init='k-medoids++', 
                        max_iter = 10000, random_state=random_state).fit(distance)
    y_km = kmedoids.labels_

    return y_km

def tsne(n_dim = 2, random_state = random_state, distance = None):
    # dim reduction
    tsne = TSNE(n_components = n_dim, metric = 'precomputed', random_state=random_state)
    tsne_embed = tsne.fit_transform(distance)

    return tsne_embed

def run_umap(n_components = 2, random_state = random_state, distance = None):
    umap_embed = umap.UMAP(n_components = n_components,
            metric = 'precomputed', random_state=random_state).fit_transform(distance)
    return umap_embed

def marker_size(tir_labels):
    min_value = np.min(tir_labels)
    max_value = np.max(tir_labels)
    func_z_list = []
    for z in tir_labels:
        if np.isnan(z):
            z = -1
        func_z_list.append(5 * (z-min_value)/(max_value - min_value))
    return func_z_list


def scatter_plot(df, tsne_embed, y_km, title, save_path):
    fig = px.scatter(
        x = tsne_embed[:,0], y = tsne_embed[:,1], 
        color=df['Group'], color_discrete_sequence= px.colors.qualitative.D3,
        opacity = 1,
        symbol = y_km[:], size = marker_size(df['AVERAGE']),
        symbol_sequence=[208,200,10, 18,20, 28, 36],
        hover_name = df.loc[:,['RBS','RBS6','Pred Mean', 'AVERAGE']].apply(
                                lambda x: ','.join(x.dropna().astype(str)),
                                axis=1
                            ),
        title = title
    )
    fig.write_html(save_path)

tsne_embed = tsne(n_dim, random_state, distance)
y_km = kmedoids(n_clusters, random_state, distance)
scatter_plot(df, tsne_embed, y_km, plot_title, save_path)

# npz_save_path = save_path[:-5] + '.npz'
# np.savez(npz_save_path, distance = distance, tsne_embed = tsne_embed, ykm = y_km)
print('result saved.')
