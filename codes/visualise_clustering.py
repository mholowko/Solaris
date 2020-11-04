
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
import os
import plotly.express as px

# Nov/2020 Mengyan Zhang
# This script visualise the DNA sequences in clusterings.
# The goal is to visualise the recommendations to see 
# whether they are similar to each other, whether they cover the whole space etc.

# Includes:
# 1) Kmediods on the pre-computed distance based on the weighted degree kernel with shift
# 2) TSNE/UMAP
# 3) Scatter plot via plotly

# read data
Folder_Path = os.getcwd() # folder path might need to change for different devices
# Path = '../../../data/Results_Microplate_partialFalse_normFalse_formatSeq_logTrue.csv'
Path = '/data/Results_Microplate_partialFalse_normTrue_roundRep_formatSeq_logTrue.csv'

df = pd.read_csv(Folder_Path + Path)
# df['Group Code'] = df.Group.astype('category').cat.codes
known_data = np.asarray(df[['RBS', 'RBS6', 'Group', 'Pred Mean', 'AVERAGE']])
known_seq = np.asarray(df['RBS'])
print('Known_seq shape ', known_seq.shape)

# setting
random_state = 24
n_dim = 2 # dimension reduction 
scores = {}

wd_shift_distance = WD_Shift_Kernel(features = known_seq, l = 6, s=1).distance_all
distance = wd_shift_distance
distance_name = 'wd_shift_distance'

n_clusters = 6 # to be changed

def kmedoids(n_clusters = 6, random_state = 0, distance = None):
    # clustering
    kmedoids = KMedoids(n_clusters=n_clusters, metric = 'precomputed', init='k-medoids++', random_state=random_state).fit(distance)
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
        func_z_list.append(5 * (z-min_value)/(max_value - min_value))
    return func_z_list


def scatter_plot(df, tsne_embed, y_km, title, save_path):
    fig = px.scatter(
        x = tsne_embed[:,0], y = tsne_embed[:,1], 
        color=df['Group'], symbol = y_km[:], size = marker_size(df['AVERAGE']),
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
plot_title = 'Round01_'+str(n_clusters)+ '_Medoids_TNSE'
save_path = Folder_Path + '/data/Clustering/' + plot_title + '.html'
scatter_plot(df, tsne_embed, y_km, plot_title, save_path)

# Save_Path = Path[:-4] + '_' + str(n_clusters) + '_medoids_' + distance_name + '.npz'
# np.savez(Folder_Path + Save_Path, ykm = y_km_spec)
# print('result saved.')
